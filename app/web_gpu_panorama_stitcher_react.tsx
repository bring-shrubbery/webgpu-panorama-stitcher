import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  ImagePlus,
  ArrowLeft,
  Download,
  Loader2,
  ChevronRight,
  Cpu,
  Layers,
  CheckCircle2,
  TriangleAlert,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

// ---- Debug helpers ----
const DEBUG = true;
function debug(...args: any[]) {
  if (DEBUG) console.log("[Panorama]", ...args);
}
function debugTimeStart(label: string) {
  if (DEBUG) console.time(`[Panorama] ${label}`);
}
function debugTimeEnd(label: string) {
  if (DEBUG) console.timeEnd(`[Panorama] ${label}`);
}

/**
 * WebGPU Panorama Stitcher
 * ------------------------------------------------------------
 * - Lets the user select multiple images
 * - Confirms selection, then stitches a panorama using WebGPU blending
 * - Uses OpenCV.js (loaded dynamically) to estimate pairwise homographies (ORB+RANSAC)
 * - Shows a progress bar while processing
 * - After completion, shows a preview with Download + Go Back
 *
 * Notes:
 * - This component assumes shadcn/ui is set up.
 * - WebGPU is used for high-speed warping & feathered blending. OpenCV.js (WASM) is used on CPU for feature matching.
 * - If OpenCV.js fails to load, a simple left-to-right paste fallback is used (no alignment).
 * - The pipeline downscales images to a configurable max dimension to keep GPU memory in check.
 */

// Let TypeScript chill about the global `cv` from OpenCV.js when it loads
declare global {
  // eslint-disable-next-line no-var
  var cv: any | undefined;
}

// Small math helpers for 3x3 matrices (column-major arrays to match WGSL expectations)
function matMul3(a: Float32Array, b: Float32Array): Float32Array {
  const r = new Float32Array(9);
  // column-major: r = a * b
  r[0] = a[0] * b[0] + a[3] * b[1] + a[6] * b[2];
  r[3] = a[0] * b[3] + a[3] * b[4] + a[6] * b[5];
  r[6] = a[0] * b[6] + a[3] * b[7] + a[6] * b[8];

  r[1] = a[1] * b[0] + a[4] * b[1] + a[7] * b[2];
  r[4] = a[1] * b[3] + a[4] * b[4] + a[7] * b[5];
  r[7] = a[1] * b[6] + a[4] * b[7] + a[7] * b[8];

  r[2] = a[2] * b[0] + a[5] * b[1] + a[8] * b[2];
  r[5] = a[2] * b[3] + a[5] * b[4] + a[8] * b[5];
  r[8] = a[2] * b[6] + a[5] * b[7] + a[8] * b[8];
  return r;
}

function matIdentity3(): Float32Array {
  const m = new Float32Array(9);
  m[0] = 1;
  m[4] = 1;
  m[8] = 1;
  return m;
}

function matTranslate(tx: number, ty: number): Float32Array {
  // column-major 3x3 translation
  const m = matIdentity3();
  m[6] = tx; // [2,0]
  m[7] = ty; // [2,1]
  return m;
}

function matInvert3(m: Float32Array): Float32Array {
  const a = m[0],
    b = m[3],
    c = m[6];
  const d = m[1],
    e = m[4],
    f = m[7];
  const g = m[2],
    h = m[5],
    i = m[8];
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-8) throw new Error("Singular matrix");
  const invDet = 1.0 / det;
  const r = new Float32Array(9);
  r[0] = A * invDet;
  r[3] = (c * h - b * i) * invDet;
  r[6] = (b * f - c * e) * invDet;
  r[1] = B * invDet;
  r[4] = (a * i - c * g) * invDet;
  r[7] = (c * d - a * f) * invDet;
  r[2] = C * invDet;
  r[5] = (b * g - a * h) * invDet;
  r[8] = (a * e - b * d) * invDet;
  return r;
}

// WGSL shader for full-frame inverse-warp + feathered alpha
const warpWGSL = /* wgsl */ `
  struct Uniforms {
    invH : mat3x3<f32>,
    srcSize : vec2<f32>,
    dstSize : vec2<f32>,
    feather : f32,
    _pad : f32, // alignment
  };
  @group(0) @binding(0) var samp : sampler;
  @group(0) @binding(1) var srcTex : texture_2d<f32>;
  @group(0) @binding(2) var<uniform> U : Uniforms;
  @group(0) @binding(3) var maskTex : texture_2d<f32>;

  struct VSOut { @builtin(position) pos : vec4<f32> };

  @vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
    // Big triangle covering the whole target
    var p = array<vec2<f32>, 3>(
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 3.0, -1.0),
      vec2<f32>(-1.0,  3.0)
    );
    var out : VSOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    return out;
  }

  @fragment fn fs_main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {
    // fragCoord.xy in pixels (0..dstSize)
    let d = vec3<f32>(fragCoord.x, fragCoord.y, 1.0);
    let s = U.invH * d; // column-major
    let w = s.z;
    if (abs(w) < 1e-5) { return vec4<f32>(0.0); }
    let uvPix = s.xy / w;

    // inside source?
    if (uvPix.x < 0.0 || uvPix.y < 0.0 || uvPix.x >= U.srcSize.x || uvPix.y >= U.srcSize.y) {
      return vec4<f32>(0.0);
    }

    // normalized UV for sampling
    let uv = (uvPix + vec2<f32>(0.5, 0.5)) / U.srcSize;
    let color = textureSampleLevel(srcTex, samp, uv, 0.0);

    // Sample seam mask in destination space (1: keep this layer, 0: suppress)
    let maskUV = (vec2<f32>(fragCoord.x, fragCoord.y) + vec2<f32>(0.5, 0.5)) / U.dstSize;
    let maskV = textureSampleLevel(maskTex, samp, maskUV, 0.0).r;
    // Use seam mask exclusively to avoid double exposure from per-image edge fading
    let alpha = clamp(maskV, 0.0, 1.0);

    // premultiply so standard alpha blending works
    return vec4<f32>(color.rgb * alpha, alpha);
  }
`;
// Robust coarse-to-fine translation via 1D profile correlation + constrained 2D NCC refine
async function estimateHomographyShift1D(
  a: ImageBitmap,
  b: ImageBitmap,
  opts?: { timeoutMs?: number; label?: string }
): Promise<Float32Array | null> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat || !cv.matchTemplate || !cv.reduce) return null;

  const timeoutMs = opts?.timeoutMs ?? 8000;
  const label = opts?.label ?? "shift1d";

  function scaleIntoGrayEdge(bmp: ImageBitmap) {
    const FEAT_MAX = 1024;
    const scale = Math.min(1, FEAT_MAX / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const rgba = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete?.();
    // Edge magnitude (Sobel) to sharpen correlation discriminativeness
    const gx = new cv.Mat(),
      gy = new cv.Mat(),
      ax = new cv.Mat(),
      ay = new cv.Mat();
    const mag = new cv.Mat();
    cv.Sobel(gray, gx, cv.CV_16S, 1, 0, 3);
    cv.Sobel(gray, gy, cv.CV_16S, 0, 1, 3);
    cv.convertScaleAbs(gx, ax);
    cv.convertScaleAbs(gy, ay);
    cv.addWeighted(ax, 0.5, ay, 0.5, 0, mag);
    gx.delete();
    gy.delete();
    ax.delete();
    ay.delete();
    return { gray, edge: mag, scale, w, h };
  }

  const work = async () => {
    debug(`${label}: start`);
    debugTimeStart(label);
    let Aedge: any, Bedge: any, Agray: any, Bgray: any;
    try {
      const {
        edge: eA,
        gray: gA,
        scale: sA,
        w: wA,
        h: hA,
      } = scaleIntoGrayEdge(a);
      const {
        edge: eB,
        gray: gB,
        scale: sB,
        w: wB,
        h: hB,
      } = scaleIntoGrayEdge(b);
      Aedge = eA;
      Bedge = eB;
      Agray = gA;
      Bgray = gB;

      // 1) Column profiles (central vertical band)
      const y0 = Math.floor(hA * 0.2),
        y1 = Math.ceil(hA * 0.8);
      const bandA = Aedge.roi(new cv.Rect(0, y0, wA, Math.max(1, y1 - y0)));
      const bandB = Bedge.roi(
        new cv.Rect(
          0,
          Math.floor(hB * 0.2),
          wB,
          Math.max(1, Math.ceil(hB * 0.8) - Math.floor(hB * 0.2))
        )
      );
      const colA = new cv.Mat();
      const colB = new cv.Mat();
      cv.reduce(bandA, colA, 0, cv.REDUCE_AVG, cv.CV_32F); // 1 x wA
      cv.reduce(bandB, colB, 0, cv.REDUCE_AVG, cv.CV_32F); // 1 x wB
      bandA.delete();
      bandB.delete();

      // Normalize columns (zero mean, unit variance) to avoid gain differences
      const norm1D = (m: any) => {
        const mm = new cv.Mat();
        const mn = new cv.Mat();
        const sd = new cv.Mat();
        cv.meanStdDev(m, mn, sd);
        const mean = mn.doubleAt(0, 0);
        const sigma = Math.max(1e-6, sd.doubleAt(0, 0));
        m.convertTo(mm, cv.CV_32F, 1.0 / sigma, -mean / sigma);
        mn.delete();
        sd.delete();
        return mm;
      };
      const colA32 = norm1D(colA);
      const colB32 = norm1D(colB);
      colA.delete();
      colB.delete();

      // 1D correlation in both directions to allow negative shifts
      const resAB = new cv.Mat();
      cv.matchTemplate(colA32, colB32, resAB, cv.TM_CCORR_NORMED); // slide B across A
      const mmAB = cv.minMaxLoc(resAB);
      const dxAB = mmAB.maxLoc.x;
      const scAB = mmAB.maxVal;

      const resBA = new cv.Mat();
      cv.matchTemplate(colB32, colA32, resBA, cv.TM_CCORR_NORMED); // slide A across B
      const mmBA = cv.minMaxLoc(resBA);
      const dxBA = mmBA.maxLoc.x;
      const scBA = mmBA.maxVal;

      resAB.delete();
      resBA.delete();
      colA32.delete();
      colB32.delete();

      // Choose direction with higher score and derive dx (B->A)
      let dx0_small: number;
      let searchDir = "B->A";
      if (scAB >= scBA) {
        dx0_small = dxAB; // placing B inside A
      } else {
        dx0_small = -dxBA; // placing A inside B ⇒ B is left of A
        searchDir = "A->B";
      }

      // 2) Refine with 2D NCC using a tall template from B, constrained around dx0
      const tW = Math.max(40, Math.floor(wB * 0.3));
      const tH = Math.max(40, Math.floor(hB * 0.6));
      const xB0 = Math.max(0, Math.min(wB - tW, Math.floor(wB * 0.35)));
      const yB0 = Math.max(0, Math.min(hB - tH, Math.floor((hB - tH) / 2)));
      const templ = Bedge.roi(new cv.Rect(xB0, yB0, tW, tH));

      // ROI in A around dx0_small with generous margin
      const marginX = Math.max(20, Math.floor(wA * 0.1));
      const cxB = xB0 + tW * 0.5;
      const cxA = cxB + dx0_small; // predicted center in A
      const xA0 = Math.max(
        0,
        Math.min(wA - tW, Math.floor(cxA - tW * 0.5 - marginX))
      );
      const xA1 = Math.max(
        xA0 + tW,
        Math.min(wA, Math.floor(cxA + tW * 0.5 + marginX))
      );
      const yA0 = Math.max(0, Math.min(hA - tH, Math.floor(hA * 0.2)));
      const roiW = Math.max(1, xA1 - xA0);
      const roiH = Math.max(1, Math.min(tH + Math.floor(hA * 0.1), hA - yA0));
      const roiA = Aedge.roi(new cv.Rect(xA0, yA0, roiW, roiH));

      const res2D = new cv.Mat();
      cv.matchTemplate(roiA, templ, res2D, cv.TM_CCOEFF_NORMED);
      const mm2 = cv.minMaxLoc(res2D);
      const best = mm2.maxLoc;
      const score2 = mm2.maxVal;

      if (!isFinite(score2) || score2 < 0.38) {
        debug(`${label}: refine score too low`, score2);
        return null;
      }

      // Subpixel refinement (quadratic) around 2D peak
      const refinePeak2D = (res: any, px: number, py: number) => {
        const cols = res.cols | 0,
          rows = res.rows | 0;
        const data = res.data32F as Float32Array;
        const idx = (y: number, x: number) => y * cols + x;
        let dx = 0,
          dy = 0;
        if (px > 0 && px < cols - 1) {
          const l = data[idx(py, px - 1)],
            c = data[idx(py, px)],
            r = data[idx(py, px + 1)];
          const den = l - 2 * c + r;
          if (Math.abs(den) > 1e-6) dx = (0.5 * (l - r)) / den;
        }
        if (py > 0 && py < rows - 1) {
          const t = data[idx(py - 1, px)],
            c = data[idx(py, px)],
            b = data[idx(py + 1, px)];
          const den = t - 2 * c + b;
          if (Math.abs(den) > 1e-6) dy = (0.5 * (t - b)) / den;
        }
        return {
          dx: Math.max(-0.75, Math.min(0.75, dx)),
          dy: Math.max(-0.75, Math.min(0.75, dy)),
        };
      };
      const sub = refinePeak2D(res2D, best.x, best.y);

      // Back to A coords
      const dx_small = xA0 + best.x + sub.dx + tW * 0.5 - (xB0 + tW * 0.5);
      const dy_small = yA0 + best.y + sub.dy + tH * 0.5 - (yB0 + tH * 0.5);

      templ.delete();
      roiA.delete();
      res2D.delete();

      // Convert to original scale and clamp vertical drift
      const maxDy = Math.max(3, Math.round(hB * 0.03));
      const dy_clamped = Math.max(-maxDy, Math.min(maxDy, dy_small));

      function mul3(a: number[], b: number[]) {
        return [
          a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
          a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
          a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
          a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
          a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
          a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
          a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
          a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
          a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
        ];
      }
      const Sa_inv = [1 / sA, 0, 0, 0, 1 / sA, 0, 0, 0, 1];
      const Sb = [sB, 0, 0, 0, sB, 0, 0, 0, 1];
      const Tsm = [1, 0, dx_small, 0, 1, dy_clamped, 0, 0, 1];
      const Hr = mul3(
        (function (a: number[], b: number[]) {
          return [
            a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
            a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
            a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
            a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
            a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
            a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
            a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
            a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
            a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
          ];
        })(Sa_inv, Tsm),
        Sb
      );

      const Hcol = new Float32Array([
        Hr[0],
        Hr[3],
        Hr[6],
        Hr[1],
        Hr[4],
        Hr[7],
        Hr[2],
        Hr[5],
        Hr[8],
      ]);
      debug(
        `${label}: dir`,
        searchDir,
        "dx0",
        dx0_small.toFixed(1),
        "refined",
        { dx: dx_small.toFixed(2), dy: dy_clamped.toFixed(2) }
      );
      debugTimeEnd(label);
      return Hcol;
    } catch (e) {
      debug(`${label}: error`, e);
      return null;
    } finally {
      try {
        Aedge?.delete?.();
        Bedge?.delete?.();
        Agray?.delete?.();
        Bgray?.delete?.();
      } catch {}
    }
  };

  try {
    return await withTimeout(work(), timeoutMs);
  } catch (e) {
    debug(`${label}: timeout`, e);
    return null;
  }
}

// Utility: dynamically load OpenCV.js once
const LOCAL_OPENCV_BASE = "/opencv/"; // served from Next.js /public/opencv/

async function validateOpenCV(): Promise<boolean> {
  const w: any = window as any;
  // Prefer cv, but fall back to Module for older OpenCV.js builds (e.g., 4.3.0)
  let cv: any = w.cv || w.Module;
  if (!cv) {
    debug("validateOpenCV: cv/Module is undefined");
    return false;
  }

  const WAIT_MS = 10000; // don't hang forever waiting for runtime

  // Wait for OpenCV runtime readiness with a timeout
  const waitForRuntime = async () => {
    // Already usable?
    if (cv.Mat) return;

    // Prefer cv.ready when present
    if (cv.ready && typeof cv.ready.then === "function") {
      debug("validateOpenCV: awaiting cv.ready");
      await Promise.race([
        cv.ready,
        new Promise((_, rej) =>
          setTimeout(() => rej(new Error("cv.ready timeout")), WAIT_MS)
        ),
      ]);
      return;
    }

    // Fallback: hook onRuntimeInitialized (chain any existing one)
    debug("validateOpenCV: waiting on onRuntimeInitialized...");
    await Promise.race([
      new Promise<void>((res) => {
        const target: any = cv; // could be Module or cv
        const prev = target.onRuntimeInitialized;
        target.onRuntimeInitialized = () => {
          debug("validateOpenCV: onRuntimeInitialized fired");
          try {
            prev?.();
          } catch {}
          res();
        };
      }),
      new Promise((_, rej) =>
        setTimeout(
          () => rej(new Error("onRuntimeInitialized timeout")),
          WAIT_MS
        )
      ),
    ]);
  };

  try {
    await waitForRuntime();
  } catch (e) {
    debug("validateOpenCV: runtime wait failed/timeout", e);
    return false;
  }

  try {
    cv = w.cv || w.Module;
    const tmp = new cv.Mat();
    tmp.delete?.();
    debug("validateOpenCV: Mat constructible ✓");
    return true;
  } catch (e) {
    debug("validateOpenCV: Mat constructible ✗", e);
    return false;
  }
}

async function loadOpenCV(): Promise<void> {
  if (typeof window === "undefined") return; // SSR guard
  const w: any = window as any;

  // If already usable, bail early
  if (await validateOpenCV()) {
    debug("loadOpenCV: already ready");
    return;
  }

  // Singleton: if another call is already loading, just await it
  if (w.__opencvPromise) {
    debug("loadOpenCV: awaiting existing loader promise");
    await w.__opencvPromise;
    return;
  }

  const INIT_TIMEOUT_MS = 20000;
  const SCRIPT_ID = "opencv-js";
  const MAX_RETRIES = 0; // avoid double-injection for older builds

  const waitForMat = async () => {
    const start = performance.now();
    while (performance.now() - start < INIT_TIMEOUT_MS) {
      if (await validateOpenCV()) return;
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error("OpenCV initialization timed out (poll)");
  };

  const preflightWasm = async (): Promise<boolean> => {
    // Deprecated: we now directly GET the wasm and validate magic bytes.
    return true;
  };

  const configureModule = () => {
    // 4.3.0 expects the global to be `Module`
    const Module = (w.Module = w.Module || {});
    if (!Module.locateFile) {
      Module.locateFile = (file: string) => `${LOCAL_OPENCV_BASE}${file}`;
    }
    return Module;
  };

  const injectScript = () => {
    let script = document.getElementById(SCRIPT_ID) as HTMLScriptElement | null;
    if (script) {
      debug("loadOpenCV: script tag already present (skip reinject)");
      return;
    }
    script = document.createElement("script");
    script.id = SCRIPT_ID;
    script.async = true;
    script.src = `${LOCAL_OPENCV_BASE}opencv.js`; // no cache-bust
    script.onload = () => debug("loadOpenCV: script onload");
    script.onerror = () => debug("loadOpenCV: script onerror");
    document.head.appendChild(script);
  };
  const attemptLoad = async (_retryIndex: number) => {
    debug(`loadOpenCV: attempt`);
    await preflightWasm();
    const Module = configureModule();

    // Try to prefetch the WASM bytes and hand them to Emscripten (works for 4.3.0 split build)
    let injectedBinary = false;
    try {
      const wasmUrl = `${LOCAL_OPENCV_BASE}opencv_js.wasm`;
      debug("loadOpenCV: fetching wasm bytes", wasmUrl);
      const wasmRes = await fetch(wasmUrl);
      const ct = wasmRes.headers.get("content-type");
      debug("loadOpenCV: GET wasm status", wasmRes.status, ct);
      if (!wasmRes.ok)
        throw new Error(`Failed to fetch wasm (${wasmRes.status})`);
      const wasmBytes = await wasmRes.arrayBuffer();
      const head = new Uint8Array(wasmBytes.slice(0, 4));
      const magicOK =
        head[0] === 0x00 &&
        head[1] === 0x61 &&
        head[2] === 0x73 &&
        head[3] === 0x6d; // "\0asm"
      if (magicOK) {
        (Module as any).wasmBinary = wasmBytes; // ensure no sync fetch path is used
        injectedBinary = true;
        debug("loadOpenCV: wasm bytes length", wasmBytes.byteLength);
      } else {
        const preview = new TextDecoder().decode(wasmBytes.slice(0, 64));
        debug("loadOpenCV: wasm magic invalid; first bytes:", head, preview);
      }
    } catch (e) {
      debug("loadOpenCV: wasm prefetch failed; continuing as single-file", e);
    }

    // Hook runtime event
    let runtimeResolved = false;
    const runtimePromise = new Promise<void>((resolve) => {
      const prev = Module.onRuntimeInitialized;
      Module.onRuntimeInitialized = () => {
        debug("loadOpenCV: onRuntimeInitialized fired");
        runtimeResolved = true;
        try {
          prev?.();
        } catch {}
        resolve();
      };
    });

    // Inject the script (no cache-bust, no reinject)
    injectScript();

    const timeout = new Promise<void>((_, rej) =>
      setTimeout(
        () => rej(new Error("OpenCV initialization timed out")),
        INIT_TIMEOUT_MS
      )
    );
    await Promise.race([runtimePromise, waitForMat(), timeout]);

    // Ensure cv alias exists for downstream code (4.3.0 exposes Module)
    if (!w.cv && w.Module) {
      w.cv = w.Module;
    }

    if (!(await validateOpenCV())) {
      if (runtimeResolved)
        debug(
          "loadOpenCV: runtime fired but validation failed (injectedBinary=",
          injectedBinary,
          ")"
        );
      throw new Error("OpenCV failed validation after init");
    }
    debug(
      "loadOpenCV: ready ✓",
      injectedBinary ? "(prefetched wasm)" : "(single-file or internal fetch)"
    );
  };

  w.__opencvPromise = (async () => {
    await attemptLoad(0);
  })();

  try {
    await w.__opencvPromise;
  } finally {
    // Keep promise for future callers
  }
}

function bitmapToMat(bitmap: ImageBitmap): any {
  const cv: any = (window as any).cv;
  if (!cv) throw new Error("OpenCV is not loaded");

  const c = document.createElement("canvas");
  c.width = bitmap.width;
  c.height = bitmap.height;

  const ctx = c.getContext("2d") as CanvasRenderingContext2D | null;
  if (!ctx) throw new Error("Canvas 2D context not available");

  ctx.drawImage(bitmap as any, 0, 0);
  const imgData = ctx.getImageData(0, 0, c.width, c.height);

  if ((cv as any).matFromImageData) {
    const m = (cv as any).matFromImageData(imgData);
    debug(
      "bitmapToMat:",
      bitmap.width,
      "x",
      bitmap.height,
      "-> Mat",
      m.cols,
      "x",
      m.rows
    );
    return m; // CV_8UC4 RGBA
  }
  const m = cv.imread(c);
  debug(
    "bitmapToMat (imread):",
    bitmap.width,
    "x",
    bitmap.height,
    "-> Mat",
    m.cols,
    "x",
    m.rows
  );
  return m;
}

// Helper: Mat -> ImageBitmap
async function matToImageBitmap(mat: any): Promise<ImageBitmap> {
  const c = document.createElement("canvas");
  c.width = mat.cols;
  c.height = mat.rows;
  cv.imshow(c, mat);
  const bmp = await createImageBitmap(c);
  return bmp;
}

// Small timeout helper so we don't hang forever on feature detection
async function withTimeout<T>(work: Promise<T>, ms: number): Promise<T> {
  return await Promise.race([
    work,
    new Promise<T>((_, rej) =>
      setTimeout(() => rej(new Error("operation timed out")), ms)
    ),
  ]);
}

// ---- Cylindrical projection (pre-warp to reduce perspective/parallax) ----
// Heuristic focal length in pixels (~60° FOV => ~0.866 * width)
function guessFocalPx(width: number, fovDeg: number = 60): number {
  const fov = (fovDeg * Math.PI) / 180;
  return (0.5 * width) / Math.tan(fov / 2);
}

function makeCylindricalMaps(w: number, h: number, f: number) {
  const cx = w * 0.5,
    cy = h * 0.5;
  // Output width in cylindrical space
  const xmin = Math.atan(-cx / f);
  const xmax = Math.atan((w - cx) / f);
  const outW = Math.max(1, Math.round(f * (xmax - xmin)));
  const outH = h; // keep same height; vertical compression handled in mapping
  const mapX = new cv.Mat(outH, outW, cv.CV_32FC1);
  const mapY = new cv.Mat(outH, outW, cv.CV_32FC1);
  for (let y = 0; y < outH; y++) {
    const Yc = (y - cy) / f;
    for (let x = 0; x < outW; x++) {
      const ang = xmin + (x / Math.max(1, outW - 1)) * (xmax - xmin); // angle around cylinder
      const X = Math.tan(ang);
      const r = Math.sqrt(1 + X * X);
      const Ys = Yc * r; // inverse of y' = f * Y / sqrt(1+X^2)
      const xs = f * X + cx;
      const ys = f * Ys + cy;
      mapX.floatPtr(y, x)[0] = xs;
      mapY.floatPtr(y, x)[0] = ys;
    }
  }
  return { mapX, mapY, outW, outH };
}

async function cylindricalProjectBitmap(
  bmp: ImageBitmap,
  fOverride?: number
): Promise<ImageBitmap> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat) return bmp; // graceful fallback
  const src = bitmapToMat(bmp);
  const f = fOverride ?? guessFocalPx(bmp.width, 60);
  const { mapX, mapY, outW, outH } = makeCylindricalMaps(
    bmp.width,
    bmp.height,
    f
  );
  const dst = new cv.Mat(outH, outW, src.type());
  cv.remap(
    src,
    dst,
    mapX,
    mapY,
    cv.INTER_LINEAR,
    cv.BORDER_CONSTANT,
    new cv.Scalar()
  );
  src.delete();
  mapX.delete();
  mapY.delete();
  const outBmp = await matToImageBitmap(dst);
  dst.delete();
  return outBmp;
}

async function cylindricalProjectAll(
  bitmaps: ImageBitmap[],
  fovDeg = 60
): Promise<ImageBitmap[]> {
  const f = guessFocalPx(bitmaps[0].width, fovDeg);
  const out: ImageBitmap[] = [];
  for (let i = 0; i < bitmaps.length; i++) {
    const b = bitmaps[i];
    // Use each image's width to compute per-image f to handle crops, but keep same FOV
    const fi = guessFocalPx(b.width, fovDeg);
    const proj = await cylindricalProjectBitmap(b, fi);
    out.push(proj);
  }
  return out;
}

// Grid-based translation score on edge maps (robust and tolerant to low texture)
async function sampleGridShiftScore(
  a: ImageBitmap,
  b: ImageBitmap,
  label: string = "fov-grid"
): Promise<null | {
  dx: number;
  dy: number;
  weight: number;
  strong: number;
  bins: number;
}> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat || !cv.matchTemplate) return null;

  function toEdge(bmp: ImageBitmap) {
    const FEAT_MAX = 1024;
    const scale = Math.min(1, FEAT_MAX / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const rgba = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete?.();
    const gx = new cv.Mat(),
      gy = new cv.Mat(),
      ax = new cv.Mat(),
      ay = new cv.Mat();
    const mag = new cv.Mat();
    cv.Sobel(gray, gx, cv.CV_16S, 1, 0, 3);
    cv.Sobel(gray, gy, cv.CV_16S, 0, 1, 3);
    cv.convertScaleAbs(gx, ax);
    cv.convertScaleAbs(gy, ay);
    cv.addWeighted(ax, 0.5, ay, 0.5, 0, mag);
    gx.delete();
    gy.delete();
    ax.delete();
    ay.delete();
    gray.delete();
    return { edge: mag, scale, w, h };
  }

  let Aedge: any, Bedge: any;
  try {
    const { edge: eA, w: wA, h: hA } = toEdge(a);
    const { edge: eB, w: wB, h: hB } = toEdge(b);
    Aedge = eA;
    Bedge = eB;

    const cols = 4,
      rows = 3;
    const tW = Math.max(24, Math.floor(wB * 0.22));
    const tH = Math.max(24, Math.floor(hB * 0.5));
    const marginX = Math.floor(wB * 0.08);
    const marginY = Math.floor(hB * 0.08);

    type Cand = { dx: number; dy: number; score: number };
    const cands: Cand[] = [];

    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const cx = Math.round(((c + 0.5) / cols) * wB);
        const cy = Math.round(((r + 0.5) / rows) * hB);
        const x0 = Math.max(
          marginX,
          Math.min(wB - marginX - tW, cx - Math.floor(tW / 2))
        );
        const y0 = Math.max(
          marginY,
          Math.min(hB - marginY - tH, cy - Math.floor(tH / 2))
        );
        const templ = Bedge.roi(new cv.Rect(x0, y0, tW, tH));
        const res = new cv.Mat();
        cv.matchTemplate(Aedge, templ, res, cv.TM_CCOEFF_NORMED);
        const mm = cv.minMaxLoc(res);
        const best = mm.maxLoc;
        const score = mm.maxVal;
        const dx_s = best.x + tW * 0.5 - (x0 + tW * 0.5);
        const dy_s = best.y + tH * 0.5 - (y0 + tH * 0.5);
        if (isFinite(score)) cands.push({ dx: dx_s, dy: dy_s, score });
        templ.delete();
        res.delete();
      }

    const maxDy = Math.max(3, Math.round(hB * 0.06));
    const strong = cands.filter(
      (c) => c.score >= 0.25 && Math.abs(c.dy) <= maxDy
    );
    if (strong.length < 3) {
      debug(label, "few strong", strong.length);
      return null;
    }

    const binW = Math.max(6, Math.round(wB * 0.035));
    const bins = new Map<number, { sumW: number; items: Cand[] }>();
    for (const c of strong) {
      const k = Math.round(c.dx / binW);
      const w = Math.max(1e-3, c.score * c.score * c.score);
      const bkt = bins.get(k) || { sumW: 0, items: [] };
      bkt.sumW += w;
      bkt.items.push(c);
      bins.set(k, bkt);
    }
    let bestK = 0,
      bestW = -1;
    bins.forEach((v, k) => {
      if (v.sumW > bestW) {
        bestW = v.sumW;
        bestK = k;
      }
    });
    const sel = strong.filter(
      (c) => Math.abs(Math.round(c.dx / binW) - bestK) <= 1
    );
    if (sel.length < 3) {
      debug(label, "no cluster");
      return null;
    }

    const sumW = sel.reduce(
      (s, c) => s + Math.max(1e-3, c.score * c.score * c.score),
      0
    );
    const dx_s =
      sel.reduce(
        (s, c) => s + c.dx * Math.max(1e-3, c.score * c.score * c.score),
        0
      ) / (sumW || 1);
    const dy_s =
      sel.reduce(
        (s, c) => s + c.dy * Math.max(1e-3, c.score * c.score * c.score),
        0
      ) / (sumW || 1);
    const dy_c = Math.max(-maxDy, Math.min(maxDy, dy_s));

    debug(label, "ok", {
      strong: strong.length,
      bins: bins.size,
      dx: dx_s.toFixed(2),
      dy: dy_c.toFixed(2),
      weight: Number(sumW.toFixed(3)),
    });
    return {
      dx: dx_s,
      dy: dy_c,
      weight: sumW,
      strong: strong.length,
      bins: bins.size,
    };
  } catch (e) {
    debug(label, "error", e);
    return null;
  } finally {
    try {
      Aedge?.delete?.();
      Bedge?.delete?.();
    } catch {}
  }
}

// Score translation between two (already pre-warped) bitmaps using edge-based NCC.
// Returns null if low confidence.
async function translationScoreEdge(
  a: ImageBitmap,
  b: ImageBitmap
): Promise<{ dx: number; dy: number; score: number } | null> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat) return null;

  function toEdge(bmp: ImageBitmap) {
    const w = bmp.width,
      h = bmp.height;
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const rgba = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete?.();
    const gx = new cv.Mat(),
      gy = new cv.Mat(),
      ax = new cv.Mat(),
      ay = new cv.Mat();
    const mag = new cv.Mat();
    cv.Sobel(gray, gx, cv.CV_16S, 1, 0, 3);
    cv.Sobel(gray, gy, cv.CV_16S, 0, 1, 3);
    cv.convertScaleAbs(gx, ax);
    cv.convertScaleAbs(gy, ay);
    cv.addWeighted(ax, 0.5, ay, 0.5, 0, mag);
    gx.delete();
    gy.delete();
    ax.delete();
    ay.delete();
    gray.delete();
    return { edge: mag, w, h };
  }

  let A: any, B: any, templ: any, res: any;
  try {
    const ea = toEdge(a);
    const eb = toEdge(b);
    A = ea.edge;
    B = eb.edge;
    // Use a tall central template from B
    const tW = Math.max(40, Math.floor(eb.w * 0.28));
    const tH = Math.max(40, Math.floor(eb.h * 0.6));
    const xB0 = Math.floor((eb.w - tW) / 2);
    const yB0 = Math.floor((eb.h - tH) / 2);
    templ = B.roi(new cv.Rect(xB0, yB0, tW, tH));

    res = new cv.Mat();
    cv.matchTemplate(A, templ, res, cv.TM_CCOEFF_NORMED);
    const mm = cv.minMaxLoc(res);
    const score = mm.maxVal;
    if (!isFinite(score) || score < 0.35) return null;
    const best = mm.maxLoc;
    const dx = best.x + tW * 0.5 - (xB0 + tW * 0.5);
    const dy = best.y + tH * 0.5 - (yB0 + tH * 0.5);
    return { dx, dy, score };
  } catch (e) {
    debug("translationScoreEdge: error", e);
    return null;
  } finally {
    try {
      templ?.delete?.();
      res?.delete?.();
      A?.delete?.();
      B?.delete?.();
    } catch {}
  }
}

// Try several FOV candidates and pick the one that yields the strongest, most horizontal alignment
async function estimateBestFOV(
  bitmaps: ImageBitmap[]
): Promise<{ fov: number; log: any }> {
  const n = bitmaps.length;
  const i0 = Math.max(0, Math.min(n - 2, Math.floor(n / 2) - 1));
  const a0 = await downscaleBitmap(bitmaps[i0], 1024);
  const b0 = await downscaleBitmap(bitmaps[i0 + 1], 1024);

  const candidates = [36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84];
  let bestFov = 60;
  let bestMetric = -Infinity;
  const logs: any[] = [];

  const evalFov = async (fov: number) => {
    const aw = await cylindricalProjectBitmap(a0, guessFocalPx(a0.width, fov));
    const bw = await cylindricalProjectBitmap(b0, guessFocalPx(b0.width, fov));
    const r = await sampleGridShiftScore(aw, bw, `autoFOV(grid,fov=${fov})`);
    try {
      (aw as any).close?.();
      (bw as any).close?.();
    } catch {}
    if (!r) {
      logs.push({ fov, result: null });
      return;
    }
    const height = Math.max(1, bw.height);
    const dyPenalty = Math.min(1, (Math.abs(r.dy) / height) * 30); // weight vertical drift strongly
    const metric = Math.log(1 + r.weight) - 0.6 * dyPenalty; // coherence minus drift
    logs.push({ fov, result: { ...r, metric } });
    if (metric > bestMetric) {
      bestMetric = metric;
      bestFov = fov;
    }
  };

  for (const f of candidates) {
    await evalFov(f);
  }

  // local refine around current best
  const start = Math.max(30, bestFov - 6),
    end = Math.min(88, bestFov + 6);
  for (let f = start; f <= end; f += 2) {
    if (candidates.includes(f)) continue;
    await evalFov(f);
  }

  if (!isFinite(bestMetric) || bestMetric === -Infinity) {
    debug("autoFOV: all candidates failed; falling back to 60");
    return { fov: 60, log: logs };
  }
  debug("autoFOV: chosen", { fov: bestFov, metric: bestMetric }, "logs", logs);
  return { fov: bestFov, log: logs };
}

// Fast translation-only estimator via normalized cross-correlation (matchTemplate)
// Now matches on edge gradients for robustness to flat textures.
async function estimateHomographyTranslation(
  a: ImageBitmap,
  b: ImageBitmap,
  opts?: { timeoutMs?: number; label?: string }
): Promise<Float32Array | null> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat) return null;

  const timeoutMs = opts?.timeoutMs ?? 6000;
  const label = opts?.label ?? "translation";

  function scaleIntoGrayEdge(bmp: ImageBitmap) {
    const FEAT_MAX = 1024;
    const scale = Math.min(1, FEAT_MAX / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const rgba = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete?.();
    // Build edge magnitude with Sobel to reduce false matches on flat textures
    const gx = new cv.Mat(),
      gy = new cv.Mat(),
      ax = new cv.Mat(),
      ay = new cv.Mat();
    const mag = new cv.Mat();
    cv.Sobel(gray, gx, cv.CV_16S, 1, 0, 3);
    cv.Sobel(gray, gy, cv.CV_16S, 0, 1, 3);
    cv.convertScaleAbs(gx, ax);
    cv.convertScaleAbs(gy, ay);
    cv.addWeighted(ax, 0.5, ay, 0.5, 0, mag);
    gx.delete();
    gy.delete();
    ax.delete();
    ay.delete();
    return { gray, edge: mag, scale, w, h };
  }

  const work = async () => {
    debug(`${label}: start`);
    debugTimeStart(label);
    let gA: any, gB: any, templ: any, result: any, grayA: any, grayB: any;
    try {
      const {
        edge: edgeA,
        gray: grayA_,
        scale: sA,
        w: wA,
        h: hA,
      } = scaleIntoGrayEdge(a);
      const {
        edge: edgeB,
        gray: grayB_,
        scale: sB,
        w: wB,
        h: hB,
      } = scaleIntoGrayEdge(b);
      gA = edgeA;
      gB = edgeB; // match on edges
      grayA = grayA_;
      grayB = grayB_;

      // Crop a central vertical strip from B as template
      const tW = Math.max(40, Math.floor(wB * 0.32));
      const tH = Math.max(40, Math.floor(hB * 0.6));
      const xB0 = Math.floor((wB - tW) / 2);
      const yB0 = Math.floor((hB - tH) / 2);
      const rectT = new cv.Rect(xB0, yB0, tW, tH);
      templ = gB.roi(rectT);

      // Slide across A (full search). Result size: (wA - tW + 1, hA - tH + 1)
      result = new cv.Mat();
      cv.matchTemplate(gA, templ, result, cv.TM_CCOEFF_NORMED);
      const mm = cv.minMaxLoc(result);
      const best = mm.maxLoc; // top-left in A where template matches best
      const score = mm.maxVal;

      if (!isFinite(score) || score < 0.35) {
        debug(`${label}: low NCC score`, score);
        return null;
      }

      // Compute shift in the downscaled space using centers
      const dx_small = best.x + tW * 0.5 - (xB0 + tW * 0.5);
      const dy_small = best.y + tH * 0.5 - (yB0 + tH * 0.5);

      // Optional clamp: small vertical drift for panorama
      const maxDy = Math.max(4, Math.round(hB * 0.06));
      const dy_small_clamped = Math.max(-maxDy, Math.min(maxDy, dy_small));

      // Scale back to original coordinates: H = S_a^-1 * T_small * S_b
      const Sa_inv = [1 / sA, 0, 0, 0, 1 / sA, 0, 0, 0, 1];
      const Sb = [sB, 0, 0, 0, sB, 0, 0, 0, 1];
      const Tsm = [1, 0, dx_small, 0, 1, dy_small_clamped, 0, 0, 1]; // row-major
      const Hr = (function mul3(a: number[], b: number[]) {
        return [
          a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
          a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
          a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
          a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
          a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
          a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
          a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
          a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
          a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
        ];
      })(
        (function mulab(a: number[], b: number[]) {
          return [
            a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
            a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
            a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
            a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
            a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
            a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
            a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
            a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
            a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
          ];
        })(Sa_inv, Tsm),
        Sb
      );

      const Hcol = new Float32Array([
        Hr[0],
        Hr[3],
        Hr[6],
        Hr[1],
        Hr[4],
        Hr[7],
        Hr[2],
        Hr[5],
        Hr[8],
      ]);
      debug(`${label}: best`, {
        score: score.toFixed(3),
        dx_small,
        dy_small,
        dy_small_clamped,
      });
      debugTimeEnd(label);
      return Hcol;
    } catch (e) {
      debug(`${label}: error`, e);
      return null;
    } finally {
      try {
        templ?.delete?.();
        result?.delete?.();
        gA?.delete?.();
        gB?.delete?.();
        grayA?.delete?.();
        grayB?.delete?.();
      } catch {}
    }
  };

  try {
    return await withTimeout(work(), timeoutMs);
  } catch (e) {
    debug(`${label}: timeout`, e);
    return null;
  }
}

// Intensity-based alignment using ECC (Enhanced Correlation Coefficient)
// Tries MOTION_TRANSLATION first; if convergence fails and images are small enough, tries MOTION_AFFINE.
async function estimateHomographyECC(
  a: ImageBitmap,
  b: ImageBitmap,
  opts?: { timeoutMs?: number; label?: string }
): Promise<Float32Array | null> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat || !cv.findTransformECC) {
    debug("ECC: findTransformECC not available");
    return null;
  }
  const timeoutMs = opts?.timeoutMs ?? 9000;
  const label = opts?.label ?? "ecc";

  function scaleIntoGray32(bmp: ImageBitmap) {
    const FEAT_MAX = 1024;
    const scale = Math.min(1, FEAT_MAX / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const rgba = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    const gray8 = new cv.Mat();
    cv.cvtColor(rgba, gray8, cv.COLOR_RGBA2GRAY);
    rgba.delete?.();
    const gray32 = new cv.Mat();
    gray8.convertTo(gray32, cv.CV_32F, 1.0 / 255.0);
    gray8.delete();
    return { gray32, scale, w, h };
  }

  const work = async () => {
    debug(`${label}: start`);
    debugTimeStart(label);
    let gA32: any, gB32: any, warp: any;
    try {
      const { gray32: A32, scale: sA, w: wA, h: hA } = scaleIntoGray32(a);
      const { gray32: B32, scale: sB, w: wB, h: hB } = scaleIntoGray32(b);
      gA32 = A32;
      gB32 = B32;

      // ECC expects warp mapping B -> A
      const criteria = new cv.TermCriteria(
        cv.TermCriteria_COUNT + cv.TermCriteria_EPS,
        80,
        1e-6
      );

      // Try pure translation first
      warp = cv.Mat.eye(2, 3, cv.CV_32F);
      let cc = 0;
      try {
        cc = cv.findTransformECC(
          gA32,
          gB32,
          warp,
          cv.MOTION_TRANSLATION,
          criteria
        );
      } catch (e) {
        debug(`${label}: translation failed`, e);
      }

      // If translation failed badly, optionally try small affine
      if (!isFinite(cc) || cc < 0.01) {
        try {
          warp = cv.Mat.eye(2, 3, cv.CV_32F);
          cc = cv.findTransformECC(
            gA32,
            gB32,
            warp,
            cv.MOTION_AFFINE,
            criteria
          );
          debug(`${label}: affine cc`, cc.toFixed ? cc.toFixed(4) : cc);
        } catch (e2) {
          debug(`${label}: affine failed`, e2);
        }
      }

      const wdat: Float32Array = warp.data32F;
      if (!wdat || wdat.length < 6) {
        debug(`${label}: warp invalid`);
        return null;
      }

      // Row-major 3x3 from 2x3
      let Hr = [wdat[0], wdat[1], wdat[2], wdat[3], wdat[4], wdat[5], 0, 0, 1];

      // Rescale back to original: H_orig = S_a^-1 * Hr * S_b
      function mul3(a: number[], b: number[]) {
        return [
          a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
          a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
          a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
          a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
          a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
          a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
          a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
          a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
          a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
        ];
      }
      function diagScale(sx: number, sy: number) {
        return [sx, 0, 0, 0, sy, 0, 0, 0, 1];
      }
      const Sa_inv = diagScale(1 / sA, 1 / sA);
      const Sb = diagScale(sB, sB);
      Hr = mul3(Sa_inv, mul3(Hr, Sb));

      // Column-major out
      const out = new Float32Array([
        Hr[0],
        Hr[3],
        Hr[6],
        Hr[1],
        Hr[4],
        Hr[7],
        Hr[2],
        Hr[5],
        Hr[8],
      ]);
      debugTimeEnd(label);
      return out;
    } catch (e) {
      debug(`${label}: error`, e);
      return null;
    } finally {
      try {
        gA32?.delete?.();
        gB32?.delete?.();
        warp?.delete?.();
      } catch {}
    }
  };

  try {
    return await withTimeout(work(), timeoutMs);
  } catch (e) {
    debug(`${label}: timeout`, e);
    return null;
  }
}

// Robust grid NCC on edge maps: many templates -> cluster dx -> weighted mean
async function estimateHomographyGridNCC(
  a: ImageBitmap,
  b: ImageBitmap,
  opts?: { timeoutMs?: number; label?: string }
): Promise<Float32Array | null> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat || !cv.matchTemplate) return null;

  const timeoutMs = opts?.timeoutMs ?? 9000;
  const label = opts?.label ?? "grid";

  function toEdge(bmp: ImageBitmap) {
    const FEAT_MAX = 1024;
    const scale = Math.min(1, FEAT_MAX / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const rgba = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete?.();
    const gx = new cv.Mat(),
      gy = new cv.Mat(),
      ax = new cv.Mat(),
      ay = new cv.Mat();
    const mag = new cv.Mat();
    cv.Sobel(gray, gx, cv.CV_16S, 1, 0, 3);
    cv.Sobel(gray, gy, cv.CV_16S, 0, 1, 3);
    cv.convertScaleAbs(gx, ax);
    cv.convertScaleAbs(gy, ay);
    cv.addWeighted(ax, 0.5, ay, 0.5, 0, mag);
    gx.delete();
    gy.delete();
    ax.delete();
    ay.delete();
    gray.delete();
    return { edge: mag, scale, w, h };
  }

  const work = async () => {
    debug(`${label}: start`);
    debugTimeStart(label);
    let Aedge: any, Bedge: any;
    try {
      const { edge: eA, scale: sA, w: wA, h: hA } = toEdge(a);
      const { edge: eB, scale: sB, w: wB, h: hB } = toEdge(b);
      Aedge = eA;
      Bedge = eB;

      const cols = 4,
        rows = 3; // sample 12 patches across B
      const tW = Math.max(40, Math.floor(wB * 0.22));
      const tH = Math.max(40, Math.floor(hB * 0.5));
      const marginX = Math.floor(wB * 0.1);
      const marginY = Math.floor(hB * 0.1);

      type Cand = { dx: number; dy: number; score: number };
      const cands: Cand[] = [];

      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const cx = Math.round(((c + 0.5) / cols) * wB);
          const cy = Math.round(((r + 0.5) / rows) * hB);
          const x0 = Math.max(
            marginX,
            Math.min(wB - marginX - tW, cx - Math.floor(tW / 2))
          );
          const y0 = Math.max(
            marginY,
            Math.min(hB - marginY - tH, cy - Math.floor(tH / 2))
          );
          const templ = Bedge.roi(new cv.Rect(x0, y0, tW, tH));
          const res = new cv.Mat();
          cv.matchTemplate(Aedge, templ, res, cv.TM_CCOEFF_NORMED);
          const mm = cv.minMaxLoc(res);
          const best = mm.maxLoc;
          const score = mm.maxVal;
          const dx_s = best.x + tW * 0.5 - (x0 + tW * 0.5);
          const dy_s = best.y + tH * 0.5 - (y0 + tH * 0.5);
          if (isFinite(score)) cands.push({ dx: dx_s, dy: dy_s, score });
          templ.delete();
          res.delete();
        }

      // filter and cluster
      const maxDy = Math.max(3, Math.round(hB * 0.06));
      const strong = cands.filter(
        (c) => c.score >= 0.35 && Math.abs(c.dy) <= maxDy
      );
      debug(`${label}: candidates`, cands.length, "strong", strong.length);
      if (strong.length < 4) return null;

      const binW = Math.max(8, Math.round(wB * 0.04)); // ~4% width
      const bins = new Map<number, { sumW: number; items: Cand[] }>();
      for (const c of strong) {
        const k = Math.round(c.dx / binW);
        const w = Math.max(1e-3, c.score * c.score * c.score);
        const bkt = bins.get(k) || { sumW: 0, items: [] };
        bkt.sumW += w;
        bkt.items.push(c);
        bins.set(k, bkt);
      }
      let bestK = 0,
        bestW = -1;
      bins.forEach((v, k) => {
        if (v.sumW > bestW) {
          bestW = v.sumW;
          bestK = k;
        }
      });
      const sel = strong.filter(
        (c) => Math.abs(Math.round(c.dx / binW) - bestK) <= 1
      );
      if (sel.length < 3) return null;

      const sumW = sel.reduce(
        (s, c) => s + Math.max(1e-3, c.score * c.score * c.score),
        0
      );
      const dx_s =
        sel.reduce(
          (s, c) => s + c.dx * Math.max(1e-3, c.score * c.score * c.score),
          0
        ) / (sumW || 1);
      const dy_s =
        sel.reduce(
          (s, c) => s + c.dy * Math.max(1e-3, c.score * c.score * c.score),
          0
        ) / (sumW || 1);
      const dy_c = Math.max(-maxDy, Math.min(maxDy, dy_s));

      // scale back to original: H = S_a^-1 * T_small * S_b
      function mul3(a: number[], b: number[]) {
        return [
          a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
          a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
          a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
          a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
          a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
          a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
          a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
          a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
          a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
        ];
      }
      const Sa_inv = [1 / sA, 0, 0, 0, 1 / sA, 0, 0, 0, 1];
      const Sb = [sB, 0, 0, 0, sB, 0, 0, 0, 1];
      const Tsm = [1, 0, dx_s, 0, 1, dy_c, 0, 0, 1];
      const Hr = mul3(
        (function (a: number[], b: number[]) {
          return [
            a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
            a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
            a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
            a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
            a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
            a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
            a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
            a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
            a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
          ];
        })(Sa_inv, Tsm),
        Sb
      );

      const Hcol = new Float32Array([
        Hr[0],
        Hr[3],
        Hr[6],
        Hr[1],
        Hr[4],
        Hr[7],
        Hr[2],
        Hr[5],
        Hr[8],
      ]);
      debug(
        `${label}: kept`,
        sel.length,
        "bins",
        bins.size,
        "dx",
        dx_s.toFixed(2),
        "dy",
        dy_c.toFixed(2)
      );
      debugTimeEnd(label);
      return Hcol;
    } catch (e) {
      debug(`${label}: error`, e);
      return null;
    } finally {
      try {
        Aedge?.delete?.();
        Bedge?.delete?.();
      } catch {}
    }
  };

  try {
    return await withTimeout(work(), timeoutMs);
  } catch (e) {
    debug(`${label}: timeout`, e);
    return null;
  }
}

// Refine a (likely-translation) homography using ECC to a small affine in the current (possibly cylindrical) space.
// Accepts column-major 3x3 H (B->A) and returns refined column-major 3x3.
async function refineAffineECCFromTranslation(
  a: ImageBitmap,
  b: ImageBitmap,
  Hcol: Float32Array,
  opts?: { timeoutMs?: number; label?: string }
): Promise<Float32Array | null> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat || !cv.findTransformECC) return null;
  const timeoutMs = opts?.timeoutMs ?? 6000;
  const label = opts?.label ?? "eccRef";

  function scaleIntoGray32(bmp: ImageBitmap) {
    const FEAT_MAX = 1024;
    const scale = Math.min(1, FEAT_MAX / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const rgba = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    const gray8 = new cv.Mat();
    cv.cvtColor(rgba, gray8, cv.COLOR_RGBA2GRAY);
    rgba.delete?.();
    const gray32 = new cv.Mat();
    gray8.convertTo(gray32, cv.CV_32F, 1.0 / 255.0);
    gray8.delete();
    return { gray32, scale, w, h };
  }
  const work = async () => {
    debug(label, "start");
    debugTimeStart(label);
    let A32: any, B32: any, warp: any;
    try {
      const { gray32: gA32, scale: sA } = scaleIntoGray32(a);
      const { gray32: gB32, scale: sB } = scaleIntoGray32(b);
      A32 = gA32;
      B32 = gB32;

      // Column-major -> row-major
      const Hrow = [
        Hcol[0],
        Hcol[3],
        Hcol[6],
        Hcol[1],
        Hcol[4],
        Hcol[7],
        Hcol[2],
        Hcol[5],
        Hcol[8],
      ];
      const diag = (sx: number, sy: number) => [sx, 0, 0, 0, sy, 0, 0, 0, 1];
      const mul3 = (a: number[], b: number[]) => [
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
      ];
      // Map original-scale H into downscaled space: H_s = S_a * H * S_b^{-1}
      const Sa = diag(sA, sA);
      const Sb_inv = diag(1 / sB, 1 / sB);
      const Hs = mul3(Sa, mul3(Hrow, Sb_inv));

      // Initialize warp (2x3) from Hs
      warp = cv.Mat.eye(2, 3, cv.CV_32F);
      const wd = warp.data32F;
      wd[0] = Hs[0];
      wd[1] = Hs[1];
      wd[2] = Hs[2];
      wd[3] = Hs[3];
      wd[4] = Hs[4];
      wd[5] = Hs[5];

      const criteria = new cv.TermCriteria(
        cv.TermCriteria_COUNT + cv.TermCriteria_EPS,
        80,
        1e-6
      );
      let cc = 0;
      try {
        cc = cv.findTransformECC(
          A32,
          B32,
          warp,
          cv.MOTION_AFFINE,
          criteria,
          null,
          5
        );
      } catch (e) {
        debug(label, "findTransformECC failed", e);
        return null;
      }
      if (!isFinite(cc) || cc < 0.01) {
        debug(label, "low cc", cc);
        return null;
      }

      // Bring back to original scale: H = S_a^{-1} * Hs_out * S_b
      const wd2 = warp.data32F;
      let Hs_out = [wd2[0], wd2[1], wd2[2], wd2[3], wd2[4], wd2[5], 0, 0, 1];
      const Sa_inv = diag(1 / sA, 1 / sA);
      const Sb = diag(sB, sB);
      const Hrow_out = mul3(Sa_inv, mul3(Hs_out, Sb));
      const Hcol_out = new Float32Array([
        Hrow_out[0],
        Hrow_out[3],
        Hrow_out[6],
        Hrow_out[1],
        Hrow_out[4],
        Hrow_out[7],
        Hrow_out[2],
        Hrow_out[5],
        Hrow_out[8],
      ]);
      debugTimeEnd(label);
      return Hcol_out;
    } catch (e) {
      debug(label, "error", e);
      return null;
    } finally {
      try {
        A32?.delete?.();
        B32?.delete?.();
        warp?.delete?.();
      } catch {}
    }
  };
  try {
    return await withTimeout(work(), timeoutMs);
  } catch (e) {
    debug(label, "timeout", e);
    return null;
  }
}

// Smart wrapper: prefer translation-like estimators after cylindrical pre-warp
async function estimatePairwiseHomography(
  a: ImageBitmap,
  b: ImageBitmap,
  opts?: { timeoutMs?: number; label?: string }
): Promise<Float32Array | null> {
  const label = opts?.label ?? "pairwise";
  // Prefer translation-like estimators after cylindrical pre-warp
  const Hg = await estimateHomographyGridNCC(a, b, {
    timeoutMs: 6500,
    label: label + "(grid)",
  });
  if (Hg) {
    const Hr = await refineAffineECCFromTranslation(a, b, Hg, {
      timeoutMs: 5000,
      label: label + "(eccRef-grid)",
    });
    return Hr || Hg;
  }
  const Hs = await estimateHomographyShift1D(a, b, {
    timeoutMs: 6000,
    label: label + "(shift1d)",
  });
  if (Hs) {
    const Hr = await refineAffineECCFromTranslation(a, b, Hs, {
      timeoutMs: 5000,
      label: label + "(eccRef-shift1d)",
    });
    return Hr || Hs;
  }
  const Ht = await estimateHomographyTranslation(a, b, {
    timeoutMs: 6000,
    label: label + "(trans)",
  });
  if (Ht) {
    const Hr = await refineAffineECCFromTranslation(a, b, Ht, {
      timeoutMs: 5000,
      label: label + "(eccRef-trans)",
    });
    return Hr || Ht;
  }
  // If none of the translation-like methods worked, try a pure ECC from scratch,
  // then corners as a last resort.
  const He = await estimateHomographyECC(a, b, {
    timeoutMs: 5000,
    label: label + "(ecc)",
  });
  if (He) return He;
  return await estimateHomographyByCorners(a, b, {
    timeoutMs: 9000,
    label: label + "(corners)",
  });
}

// Fallback homography when ORB/AKAZE/BRISK are unavailable (e.g., OpenCV.js 4.3.0 without features2d)
// Strategy: detect Shi-Tomasi corners in B, extract small patches, search a local ROI in A via matchTemplate (TM_CCOEFF_NORMED),
// then RANSAC a homography from the matched pairs. Works best for panoramas with mostly-horizontal motion.
async function estimateHomographyByCorners(
  a: ImageBitmap, // previous image (target)
  b: ImageBitmap, // current image (source)
  opts?: { timeoutMs?: number; label?: string }
): Promise<Float32Array | null> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat) {
    debug("estimateHomographyByCorners: OpenCV not ready – skipping");
    return null;
  }

  const timeoutMs = opts?.timeoutMs ?? 12000;
  const label = (opts?.label ?? "corner-homography") + "(corners)";

  const FEAT_MAX = 800; // smaller for speed since we do many templates
  const MAX_BPTS = 90; // corners to sample from B
  const PATCH = 25; // patch size (odd)
  const HALF = (PATCH - 1) / 2;
  const MAX_DX = Math.round(FEAT_MAX * 0.18); // search ~18% width
  const MAX_DY = Math.round(FEAT_MAX * 0.06); // small vertical drift
  const THRESH = 0.78; // NCC score threshold

  function scaleIntoGray(bmp: ImageBitmap) {
    const scale = Math.min(1, FEAT_MAX / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const rgba = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete?.();
    return { gray, scale, w, h };
  }

  const work = async () => {
    debug(label, "start");
    debugTimeStart(label);
    let gA: any,
      gB: any,
      cornersB: any,
      patchB: any,
      roiA: any,
      result: any,
      mask: any;
    try {
      const { gray: grayA, scale: sA, w: wA, h: hA } = scaleIntoGray(a);
      const { gray: grayB, scale: sB, w: wB, h: hB } = scaleIntoGray(b);
      gA = grayA;
      gB = grayB;
      debug(label, "sizes", { A: [wA, hA], B: [wB, hB] }, "scales", { sA, sB });

      // Detect corners on B
      cornersB = new cv.Mat();
      cv.goodFeaturesToTrack(gB, cornersB, MAX_BPTS, 0.01, 8);
      const nb = cornersB.rows; // Nx1 CV_32FC2
      if (!nb || nb < 8) {
        debug(label, "no corners on B");
        return null;
      }

      const srcPts: number[] = []; // B
      const dstPts: number[] = []; // A

      // Pre-allocs
      patchB = new cv.Mat();
      result = new cv.Mat();

      const clamp = (v: number, min: number, max: number) =>
        Math.max(min, Math.min(max, v));

      for (let i = 0; i < nb; i++) {
        const xB = cornersB.data32F[i * 2];
        const yB = cornersB.data32F[i * 2 + 1];
        if (xB < HALF || yB < HALF || xB >= wB - HALF || yB >= hB - HALF)
          continue;

        // Extract integer-aligned patch around (xB,yB) in B (fallback for builds without getRectSubPix)
        const cx = Math.round(xB);
        const cy = Math.round(yB);
        const rx = cx - HALF;
        const ry = cy - HALF;
        if (rx < 0 || ry < 0 || rx + PATCH > wB || ry + PATCH > hB) {
          continue;
        }
        const rectB = new cv.Rect(rx, ry, PATCH, PATCH);
        const roiB = gB.roi(rectB);
        roiB.copyTo(patchB);
        roiB.delete();

        // Build search ROI in A
        const x0 = clamp(Math.round(xB - MAX_DX - HALF), 0, wA - PATCH);
        const y0 = clamp(Math.round(yB - MAX_DY - HALF), 0, hA - PATCH);
        const x1 = clamp(Math.round(xB + MAX_DX + HALF), 0, wA - 1);
        const y1 = clamp(Math.round(yB + MAX_DY + HALF), 0, hA - 1);
        const roiW = x1 - x0 + 1;
        const roiH = y1 - y0 + 1;
        if (roiW < PATCH || roiH < PATCH) continue;

        const rect = new cv.Rect(x0, y0, roiW, roiH);
        roiA = gA.roi(rect);

        // matchTemplate (same as sliding normalized correlation) in A with patch from B
        cv.matchTemplate(roiA, patchB, result, cv.TM_CCOEFF_NORMED);
        const mm = cv.minMaxLoc(result);
        const maxVal = mm.maxVal;
        if (maxVal < THRESH) {
          roiA.delete();
          continue;
        }

        const best = mm.maxLoc; // top-left within roiA
        const xA = x0 + best.x + HALF;
        const yA = y0 + best.y + HALF;

        // Save pair (B->A)
        srcPts.push(xB, yB);
        dstPts.push(xA, yA);

        roiA.delete();
      }

      debug(label, "candidate pairs", srcPts.length / 2);
      if (srcPts.length < 16) {
        debug(label, "insufficient pairs");
        return null;
      }

      // Build Mats for findHomography
      const n = srcPts.length / 2;
      const ptsB = cv.Mat.zeros(n, 1, cv.CV_32FC2);
      const ptsA = cv.Mat.zeros(n, 1, cv.CV_32FC2);
      for (let i = 0; i < n; i++) {
        ptsB.data32F[i * 2] = srcPts[i * 2];
        ptsB.data32F[i * 2 + 1] = srcPts[i * 2 + 1];
        ptsA.data32F[i * 2] = dstPts[i * 2];
        ptsA.data32F[i * 2 + 1] = dstPts[i * 2 + 1];
      }

      mask = new cv.Mat();
      const H = cv.findHomography(ptsB, ptsA, cv.RANSAC, 3.0, mask);
      if (!H || H.empty()) {
        debug(label, "findHomography returned empty");
        return null;
      }

      // Log inliers
      try {
        let inliers = 0;
        for (let i = 0; i < mask.rows; i++) {
          if (mask.ucharPtr(i, 0)[0]) inliers++;
        }
        debug(label, "inliers", `${inliers}/${n}`);
      } catch {}

      // Convert to original scale: H_orig = S_a^-1 * H_small * S_b
      const h = H.data64F?.length ? H.data64F : H.data32F; // row-major
      let Hr = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]];
      function mul3(a: number[], b: number[]) {
        return [
          a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
          a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
          a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
          a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
          a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
          a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
          a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
          a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
          a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
        ];
      }
      function diagScale(sx: number, sy: number) {
        return [sx, 0, 0, 0, sy, 0, 0, 0, 1];
      }
      const Sa_inv = diagScale(1 / sA, 1 / sA);
      const Sb = diagScale(sB, sB);
      Hr = mul3(Sa_inv, mul3(Hr, Sb));

      debugTimeEnd(label);
      return new Float32Array([
        Hr[0],
        Hr[3],
        Hr[6],
        Hr[1],
        Hr[4],
        Hr[7],
        Hr[2],
        Hr[5],
        Hr[8],
      ]);
    } catch (e) {
      debug(label, "error", e);
      return null;
    } finally {
      try {
        gA?.delete?.();
        gB?.delete?.();
        cornersB?.delete?.();
        patchB?.delete?.();
        result?.delete?.();
        roiA?.delete?.();
        mask?.delete?.();
      } catch {}
    }
  };

  try {
    return await withTimeout(work(), timeoutMs);
  } catch (e) {
    debug(label, "timeout or error", e);
    return null;
  }
}

// Estimate pairwise homography from current image B -> previous image A using ORB/AKAZE/BRISK + RANSAC
// Uses downscaled copies for speed and rescales the homography back to original resolution.
async function estimateHomographyORB(
  a: ImageBitmap,
  b: ImageBitmap,
  opts?: { timeoutMs?: number; label?: string }
): Promise<Float32Array | null> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat) {
    debug("estimateHomographyORB: OpenCV not ready – skipping");
    return null;
  }

  const timeoutMs = opts?.timeoutMs ?? 12000;
  const label = opts?.label ?? "homography";
  const FEAT_MAX = 1024; // max side in pixels for feature detection

  function scaleIntoMat(bmp: ImageBitmap) {
    const scale = Math.min(1, FEAT_MAX / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(bmp as any, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const mat = cv.matFromImageData
      ? cv.matFromImageData(imgData)
      : cv.imread(c);
    return { mat, scale };
  }

  function mul3(a: number[], b: number[]) {
    // row-major 3x3 * 3x3
    return [
      a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
      a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
      a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
      a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
      a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
      a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
      a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
      a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
      a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ];
  }

  function diagScale(sx: number, sy: number): number[] {
    // row-major
    return [sx, 0, 0, 0, sy, 0, 0, 0, 1];
  }

  const work = async () => {
    debug(label, "start");
    debugTimeStart(label);
    let imgA: any,
      imgB: any,
      grayA: any,
      grayB: any,
      kpA: any,
      kpB: any,
      desA: any,
      desB: any,
      matches: any,
      bf: any,
      mask: any;
    try {
      const { mat: srcA, scale: sA } = scaleIntoMat(a);
      const { mat: srcB, scale: sB } = scaleIntoMat(b);
      imgA = srcA;
      imgB = srcB;
      debug(label, "scales", { sA, sB }, "sizes", {
        A: [imgA.cols, imgA.rows],
        B: [imgB.cols, imgB.rows],
      });

      grayA = new cv.Mat();
      grayB = new cv.Mat();
      cv.cvtColor(imgA, grayA, cv.COLOR_RGBA2GRAY);
      cv.cvtColor(imgB, grayB, cv.COLOR_RGBA2GRAY);

      let detector: any = null;
      if (cv.ORB && cv.ORB.create) detector = cv.ORB.create(1200);
      else if (cv.AKAZE && cv.AKAZE.create) detector = cv.AKAZE.create();
      else if (cv.BRISK && cv.BRISK.create) detector = cv.BRISK.create();
      if (!detector) {
        debug(label, "no detector available – falling back to corner matcher");
        // Clean up local mats before delegating
        imgA?.delete?.();
        imgB?.delete?.();
        grayA?.delete?.();
        grayB?.delete?.();
        // Use the corner-based fallback at the same downscaled resolution
        return await estimateHomographyByCorners(a, b, { timeoutMs, label });
      }

      kpA = new cv.KeyPointVector();
      kpB = new cv.KeyPointVector();
      desA = new cv.Mat();
      desB = new cv.Mat();
      detector.detectAndCompute(grayA, new cv.Mat(), kpA, desA);
      detector.detectAndCompute(grayB, new cv.Mat(), kpB, desB);
      const kpCountA = typeof kpA?.size === "function" ? kpA.size() : 0;
      const kpCountB = typeof kpB?.size === "function" ? kpB.size() : 0;
      debug(label, "keypoints", { A: kpCountA, B: kpCountB }, "des", {
        A: desA.rows,
        B: desB.rows,
      });
      if (desA.rows === 0 || desB.rows === 0) {
        debug(label, "no descriptors");
        return null;
      }

      const norm = cv.NORM_HAMMING;
      bf = new cv.BFMatcher(norm, false);
      matches = new cv.DMatchVectorVector();
      bf.knnMatch(desB, desA, matches, 2); // B -> A
      const matchCount =
        typeof matches.size === "function" ? matches.size() : 0;
      debug(label, "matches", matchCount);

      const good: any[] = [];
      for (let i = 0; i < matchCount; i++) {
        const m = matches.get(i).get(0);
        const n = matches.get(i).get(1);
        if (m && n && m.distance < 0.75 * n.distance) good.push(m);
      }
      debug(label, "good matches", good.length);
      if (good.length < 8) {
        debug(label, "insufficient good matches");
        return null;
      }

      const ptsB = new cv.Mat(good.length, 1, cv.CV_32FC2);
      const ptsA = new cv.Mat(good.length, 1, cv.CV_32FC2);
      for (let i = 0; i < good.length; i++) {
        const pB = kpB.get(good[i].queryIdx).pt; // B
        const pA = kpA.get(good[i].trainIdx).pt; // A
        ptsB.data32F[i * 2] = pB.x;
        ptsB.data32F[i * 2 + 1] = pB.y;
        ptsA.data32F[i * 2] = pA.x;
        ptsA.data32F[i * 2 + 1] = pA.y;
      }

      mask = new cv.Mat();
      const H = cv.findHomography(ptsB, ptsA, cv.RANSAC, 3.0, mask);
      if (!H || H.empty()) {
        debug(label, "findHomography returned empty");
        return null;
      }

      // Count inliers from mask, if available
      try {
        let inliers = 0;
        for (let i = 0; i < mask.rows; i++) {
          if (mask.ucharPtr(i, 0)[0]) inliers++;
        }
        debug(label, "inliers", `${inliers}/${good.length}`);
      } catch {}

      // Row-major H_small
      const h = H.data64F?.length ? H.data64F : H.data32F;
      let Hr = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]];
      // Rescale back to original coordinates: H_orig = S_a^-1 * H_small * S_b
      const Sa_inv = diagScale(1 / sA, 1 / sA);
      const Sb = diagScale(sB, sB);
      Hr = mul3(Sa_inv, mul3(Hr, Sb));

      debugTimeEnd(label);
      // Convert row-major -> column-major Float32Array
      const out = new Float32Array([
        Hr[0],
        Hr[3],
        Hr[6],
        Hr[1],
        Hr[4],
        Hr[7],
        Hr[2],
        Hr[5],
        Hr[8],
      ]);
      return out;
    } catch (e) {
      debug(label, "error", e);
      throw e;
    } finally {
      try {
        imgA?.delete?.();
        imgB?.delete?.();
        grayA?.delete?.();
        grayB?.delete?.();
        kpA?.delete?.();
        kpB?.delete?.();
        desA?.delete?.();
        desB?.delete?.();
        matches?.delete?.();
        mask?.delete?.();
        bf?.delete?.();
      } catch {}
    }
  };

  try {
    return await withTimeout(work(), timeoutMs);
  } catch (e) {
    debug(label, "timeout or error", e);
    return null;
  }
}

// Compute cumulative transforms to map each image i into the reference (image 0) frame.
async function computeCumulativeHomographies(
  bitmaps: ImageBitmap[],
  onProgress?: (p: number) => void
): Promise<(Float32Array | null)[]> {
  debug("computeCumulativeHomographies: count", bitmaps.length);
  const n = bitmaps.length;
  const H: (Float32Array | null)[] = new Array(n).fill(null);
  H[0] = matIdentity3();
  for (let i = 1; i < n; i++) {
    const label = `H[${i - 1}->${i}]`;
    onProgress?.(i / n);
    debug(label, "estimating");
    const t0 = performance.now();
    const Hij = await estimatePairwiseHomography(bitmaps[i - 1], bitmaps[i], {
      timeoutMs: 12000,
      label,
    });
    const dt = (performance.now() - t0).toFixed(1);
    if (!Hij) {
      debug(label, `FAILED in ${dt}ms`);
      H[i] = null;
      continue;
    }
    debug(label, `ok in ${dt}ms`);
    debug(
      label,
      "H_ij",
      Array.from(Hij as Float32Array).map((v) => Number(v.toFixed(3)))
    );
    H[i] = matMul3(H[i - 1]!, Hij); // compose toward frame 0
  }
  onProgress?.(1);
  return H;
}

// Create a GPUTexture from ImageBitmap
function createTextureFromBitmap(
  device: GPUDevice,
  bitmap: ImageBitmap
): GPUTexture {
  const texture = device.createTexture({
    size: { width: bitmap.width, height: bitmap.height },
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  // @ts-ignore
  device.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture },
    { width: bitmap.width, height: bitmap.height }
  );
  return texture;
}

// Read a color texture into ImageData (handles 256 BPR padding)
async function readTextureToImageData(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number
): Promise<ImageData> {
  const bytesPerPixel = 4;
  const align = 256;
  const bytesPerRow = Math.ceil((width * bytesPerPixel) / align) * align;
  const size = bytesPerRow * height;
  const buffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer, bytesPerRow },
    { width, height, depthOrArrayLayers: 1 }
  );
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buffer.getMappedRange());
  const pixels = new Uint8ClampedArray(width * height * bytesPerPixel);

  for (let y = 0; y < height; y++) {
    const srcStart = y * bytesPerRow;
    const dstStart = y * width * bytesPerPixel;
    pixels.set(
      mapped.subarray(srcStart, srcStart + width * bytesPerPixel),
      dstStart
    );
  }
  buffer.unmap();
  return new ImageData(pixels, width, height);
}

async function ensureWebGPU(): Promise<GPUDevice> {
  if (!("gpu" in navigator))
    throw new Error("WebGPU not supported in this browser");
  const adapter = await (navigator as any).gpu.requestAdapter();
  if (!adapter) throw new Error("Failed to get GPU adapter");
  const device = await adapter.requestDevice();
  return device;
}

// Downscale an ImageBitmap if it's bigger than maxDim
async function downscaleBitmap(
  bitmap: ImageBitmap,
  maxDim: number
): Promise<ImageBitmap> {
  const w = bitmap.width,
    h = bitmap.height;
  const scale = Math.min(1, maxDim / Math.max(w, h));
  if (scale >= 0.999) return bitmap;
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w * scale));
  c.height = Math.max(1, Math.round(h * scale));
  const ctx = c.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, c.width, c.height);
  const out = await createImageBitmap(c);
  return out;
}

// Compute panorama extents after applying H[i] to each image's corners.
function computePanoramaBounds(
  bitmaps: ImageBitmap[],
  transforms: (Float32Array | null)[]
) {
  const pts: { x: number; y: number }[] = [];
  function apply(m: Float32Array, x: number, y: number) {
    const X = m[0] * x + m[3] * y + m[6];
    const Y = m[1] * x + m[4] * y + m[7];
    const W = m[2] * x + m[5] * y + m[8];
    return { x: X / W, y: Y / W };
  }
  for (let i = 0; i < bitmaps.length; i++) {
    const H = transforms[i];
    if (!H) continue;
    const w = bitmaps[i].width,
      h = bitmaps[i].height;
    pts.push(apply(H, 0, 0));
    pts.push(apply(H, w, 0));
    pts.push(apply(H, 0, h));
    pts.push(apply(H, w, h));
  }
  if (pts.length === 0) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  return {
    minX: Math.floor(Math.min(...xs)),
    minY: Math.floor(Math.min(...ys)),
    maxX: Math.ceil(Math.max(...xs)),
    maxY: Math.ceil(Math.max(...ys)),
  };
}

// Naive CPU fallback: paste images left-to-right without blending
async function naiveStitch(bitmaps: ImageBitmap[]): Promise<string> {
  const totalWidth = bitmaps.reduce((s, b) => s + b.width, 0);
  const maxH = Math.max(...bitmaps.map((b) => b.height));
  const c = document.createElement("canvas");
  c.width = totalWidth;
  c.height = maxH;
  const ctx = c.getContext("2d")!;
  let x = 0;
  for (const b of bitmaps) {
    ctx.drawImage(b, x, 0);
    x += b.width;
  }
  debug("naiveStitch: composed", { totalWidth, maxH, count: bitmaps.length });
  return c.toDataURL("image/png");
}

// --- Seam mask builder (OpenCV) -------------------------------------------------
// Build per-image destination-space masks that cut a minimal-error vertical seam
// through the overlap with the previous image. The mask value is 1 where the
// *current* image should show, 0 where it should give way to the previous one.
async function buildSeamMasks(
  bitmaps: ImageBitmap[],
  finalH: Float32Array[], // column-major source->panorama transforms
  dstW: number,
  dstH: number,
  feather: number,
  seamWidthPx?: number // explicit seam width in DESTINATION pixels
): Promise<(ImageBitmap | null)[]> {
  await loadOpenCV();
  const cv: any = (window as any).cv;
  if (!cv || !cv.Mat) {
    debug("buildSeamMasks: OpenCV not ready – skipping");
    return new Array(bitmaps.length).fill(null);
  }

  // Column-major -> row-major
  const colToRow = (m: Float32Array): number[] => [
    m[0],
    m[1],
    m[2],
    m[3],
    m[4],
    m[5],
    m[6],
    m[7],
    m[8],
  ];
  // Row-major 3x3 multiply
  const mulR = (a: number[], b: number[]) => [
    a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
    a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
    a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
    a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
    a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
    a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
    a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
    a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
    a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
  ];
  const applyR = (H: number[], x: number, y: number) => {
    const X = H[0] * x + H[1] * y + H[2];
    const Y = H[3] * x + H[4] * y + H[5];
    const W = H[6] * x + H[7] * y + H[8];
    return { x: X / W, y: Y / W };
  };
  const rectOf = (H: number[], w: number, h: number) => {
    const p = [
      applyR(H, 0, 0),
      applyR(H, w, 0),
      applyR(H, 0, h),
      applyR(H, w, h),
    ];
    const xs = p.map((p) => p.x),
      ys = p.map((p) => p.y);
    return {
      minX: Math.floor(Math.min(...xs)),
      minY: Math.floor(Math.min(...ys)),
      maxX: Math.ceil(Math.max(...xs)),
      maxY: Math.ceil(Math.max(...ys)),
    };
  };
  const intersect = (a: any, b: any) => {
    const minX = Math.max(a.minX, b.minX);
    const minY = Math.max(a.minY, b.minY);
    const maxX = Math.min(a.maxX, b.maxX);
    const maxY = Math.min(a.maxY, b.maxY);
    if (maxX <= minX || maxY <= minY) return null;
    return { minX, minY, maxX, maxY };
  };

  const masks: (ImageBitmap | null)[] = new Array(bitmaps.length).fill(null);
  const seamScale = 0.35; // build seam on a reduced grid for speed

  for (let i = 1; i < bitmaps.length; i++) {
    const A = bitmaps[i - 1];
    const B = bitmaps[i];
    const HAr = colToRow(finalH[i - 1]);
    const HBr = colToRow(finalH[i]);

    const rectA = rectOf(HAr, A.width, A.height);
    const rectB = rectOf(HBr, B.width, B.height);
    const ov = intersect(rectA, rectB);
    if (!ov) {
      debug("seam: no overlap for", i - 1, "->", i);
      continue;
    }
    const ovW = ov.maxX - ov.minX,
      ovH = ov.maxY - ov.minY;
    if (ovW < 8 || ovH < 8) {
      debug("seam: tiny overlap for", i);
      continue;
    }

    // ROI mapping from panorama -> seam grid
    const ow = Math.max(16, Math.floor(ovW * seamScale));
    const oh = Math.max(16, Math.floor(ovH * seamScale));
    const sx = ow / ovW,
      sy = oh / ovH;
    const R = [sx, 0, -sx * ov.minX, 0, sy, -sy * ov.minY, 0, 0, 1];
    const H_A_roi = mulR(R, HAr);
    const H_B_roi = mulR(R, HBr);

    // Warp both into ROI grid (grayscale)
    const srcA = bitmapToMat(A);
    const srcB = bitmapToMat(B);
    const gA = new cv.Mat();
    const gB = new cv.Mat();
    cv.cvtColor(srcA, gA, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(srcB, gB, cv.COLOR_RGBA2GRAY);
    srcA.delete();
    srcB.delete();
    const size = new cv.Size(ow, oh);
    const wA = new cv.Mat(oh, ow, cv.CV_8U);
    const wB = new cv.Mat(oh, ow, cv.CV_8U);
    const H_A_cv = cv.matFromArray(3, 3, cv.CV_64F, H_A_roi);
    const H_B_cv = cv.matFromArray(3, 3, cv.CV_64F, H_B_roi);
    cv.warpPerspective(
      gA,
      wA,
      H_A_cv,
      size,
      cv.INTER_LINEAR,
      cv.BORDER_CONSTANT,
      new cv.Scalar()
    );
    cv.warpPerspective(
      gB,
      wB,
      H_B_cv,
      size,
      cv.INTER_LINEAR,
      cv.BORDER_CONSTANT,
      new cv.Scalar()
    );
    gA.delete();
    gB.delete();
    H_A_cv.delete();
    H_B_cv.delete();

    // Cost map on edge magnitudes (reduces texture-based false positives)
    const sobelMag = (img: any) => {
      const gx = new cv.Mat(),
        gy = new cv.Mat(),
        ax = new cv.Mat(),
        ay = new cv.Mat(),
        mag = new cv.Mat();
      cv.Sobel(img, gx, cv.CV_16S, 1, 0, 3);
      cv.Sobel(img, gy, cv.CV_16S, 0, 1, 3);
      cv.convertScaleAbs(gx, ax);
      cv.convertScaleAbs(gy, ay);
      cv.addWeighted(ax, 0.5, ay, 0.5, 0, mag);
      gx.delete();
      gy.delete();
      ax.delete();
      ay.delete();
      return mag;
    };
    const eA = sobelMag(wA),
      eB = sobelMag(wB);
    const diff = new cv.Mat();
    cv.absdiff(eA, eB, diff);
    eA.delete();
    eB.delete();
    cv.blur(diff, diff, new cv.Size(5, 5));

    // DP minimal vertical path (top -> bottom)
    const W = ow,
      Hh = oh;
    const d: Uint8Array = diff.data;
    const cost = new Float32Array(W * Hh);
    const back = new Int16Array(W * Hh);
    for (let x = 0; x < W; x++) {
      cost[x] = d[x];
      back[x] = -1;
    }
    for (let y = 1; y < Hh; y++) {
      const off = y * W,
        prev = (y - 1) * W;
      for (let x = 0; x < W; x++) {
        let bx = x,
          bc = cost[prev + x];
        if (x > 0 && cost[prev + x - 1] < bc) {
          bc = cost[prev + x - 1];
          bx = x - 1;
        }
        if (x < W - 1 && cost[prev + x + 1] < bc) {
          bc = cost[prev + x + 1];
          bx = x + 1;
        }
        cost[off + x] = d[off + x] + bc;
        back[off + x] = bx;
      }
    }
    let endX = 0,
      best = Infinity;
    const last = (Hh - 1) * W;
    for (let x = 0; x < W; x++) {
      const v = cost[last + x];
      if (v < best) {
        best = v;
        endX = x;
      }
    }
    const seam = new Int16Array(Hh);
    seam[Hh - 1] = endX;
    for (let y = Hh - 1; y > 0; y--) {
      seam[y - 1] = back[y * W + seam[y]];
    }
    diff.delete();
    wA.delete();
    wB.delete();

    // Decide which side to keep for image B (usually to the right if centers increase)
    const centerA = applyR(HAr, A.width * 0.5, A.height * 0.5);
    const centerB = applyR(HBr, B.width * 0.5, B.height * 0.5);
    const keepRight = centerB.x >= centerA.x;

    // Build ROI mask via per-row smooth ramp around the seam (soft crossfade)
    // The mask value is 1 on the kept side and 0 on the suppressed side.
    // We specify fade width in **destination pixels** and convert to ROI pixels here.
    const roiCanvas = document.createElement("canvas");
    roiCanvas.width = ow;
    roiCanvas.height = oh;
    const rctx = roiCanvas.getContext("2d")!;
    const id = rctx.createImageData(ow, oh);
    const px = id.data; // RGBA

    // Convert desired fade width in destination space -> ROI space
    const desiredDestFade = Math.max(2, Math.round(seamWidthPx ?? feather)); // px in the final panorama
    const scaleX = ow / Math.max(1, ovW); // ROI pixels per 1 destination pixel across X
    const fadeW = Math.max(2, Math.round(desiredDestFade * scaleX));
    const twoW = 2 * fadeW;

    const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
    const smooth = (t: number) => {
      t = clamp01(t);
      return t * t * (3 - 2 * t);
    }; // cubic smoothstep

    for (let y = 0; y < oh; y++) {
      const s = Math.max(0, Math.min(ow - 1, seam[y]));
      const rampStart = s - fadeW; // where ramp begins
      for (let x = 0; x < ow; x++) {
        // normalize x across the 2*fadeW window centered at seam
        const t = (x - rampStart) / Math.max(1, twoW); // 0 at start, 1 at end
        const base = smooth(t);
        const a = keepRight ? base : 1 - base; // 0..1
        const v = Math.max(0, Math.min(255, Math.round(a * 255)));
        const k = (y * ow + x) * 4;
        px[k] = v;
        px[k + 1] = v;
        px[k + 2] = v;
        px[k + 3] = 255;
      }
    }
    rctx.putImageData(id, 0, 0);
    debug("seam: ramp", {
      desiredDestFade,
      fadeW_roi: fadeW,
      ow,
      oh,
      ovW,
      ovH,
    });

    // Compose into a panorama-sized mask
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = dstW;
    maskCanvas.height = dstH;
    const mctx = maskCanvas.getContext("2d")!;

    // Fill the whole polygon footprint of image B with 1s so non-overlap stays fully on
    const applyCM = (H: number[], x: number, y: number) => {
      const X = H[0] * x + H[1] * y + H[2];
      const Y = H[3] * x + H[4] * y + H[5];
      const Ww = H[6] * x + H[7] * y + H[8];
      return { x: X / Ww, y: Y / Ww };
    };
    const p0 = applyCM(HBr, 0, 0),
      p1 = applyCM(HBr, B.width, 0),
      p2 = applyCM(HBr, B.width, B.height),
      p3 = applyCM(HBr, 0, B.height);
    mctx.fillStyle = "#fff";
    mctx.beginPath();
    mctx.moveTo(p0.x, p0.y);
    mctx.lineTo(p1.x, p1.y);
    mctx.lineTo(p2.x, p2.y);
    mctx.lineTo(p3.x, p3.y);
    mctx.closePath();
    mctx.fill();

    // Paste the ramp only inside the overlap rectangle
    mctx.save();
    mctx.beginPath();
    mctx.rect(ov.minX, ov.minY, ovW, ovH);
    mctx.clip();
    mctx.imageSmoothingEnabled = true;
    mctx.imageSmoothingQuality = "high";
    mctx.drawImage(roiCanvas, ov.minX, ov.minY, ovW, ovH);
    mctx.restore();

    const maskBmp = await createImageBitmap(maskCanvas);
    masks[i] = maskBmp;
    debug("seam: built mask for", i, "overlap", { w: ovW, h: ovH });
  }

  return masks;
}

// Render stitched panorama with WebGPU. Returns a data URL.
async function stitchWithWebGPU(
  bitmaps: ImageBitmap[],
  feather = 60,
  seamWidthPx = 0,
  setProgress?: (p: number) => void
): Promise<string> {
  setProgress?.(5);
  debug("stitch: ensuring WebGPU");
  const device = await ensureWebGPU();
  debug("stitch: WebGPU ready");
  setProgress?.(8);

  // 1) Load OpenCV + compute homographies
  try {
    await loadOpenCV();
  } catch (e) {
    debug("stitch: OpenCV load error", e);
  }
  setProgress?.(12);

  // Auto-detect FOV on a representative pair, then pre-warp
  debug("stitch: estimating FOV");
  const { fov } = await estimateBestFOV(bitmaps);
  debug("stitch: cylindrical pre-warp fov", fov);
  const cylBitmaps = await cylindricalProjectAll(bitmaps, fov);

  debug("stitch: computing homographies");
  const homos = await computeCumulativeHomographies(cylBitmaps, (p) => {
    const val = 12 + Math.round(p * 38);
    debug("progress:", val);
    setProgress?.(val);
  });

  // If any homography missing, fallback entirely
  if (homos.some((h) => h === null)) {
    debug("stitch: homography missing for at least one pair -> naive fallback");
    return await naiveStitch(cylBitmaps);
  }

  // 2) Compute panorama bounds, translation to keep positive
  const bounds = computePanoramaBounds(cylBitmaps, homos as Float32Array[]);
  const panWidth = Math.max(1, bounds.maxX - bounds.minX);
  const panHeight = Math.max(1, bounds.maxY - bounds.minY);
  debug("stitch: bounds", bounds, "pan", { panWidth, panHeight });

  // Limit huge panoramas to something sane (8K width cap)
  const MAX_W = 8192,
    MAX_H = 4096;
  let scale = Math.min(MAX_W / panWidth, MAX_H / panHeight, 1);
  const dstW = Math.max(1, Math.floor(panWidth * scale));
  const dstH = Math.max(1, Math.floor(panHeight * scale));
  debug("stitch: dst size", { dstW, dstH, scale });

  const T = matTranslate(-bounds.minX, -bounds.minY); // translate world -> +ve
  const S = new Float32Array([
    // scale
    scale,
    0,
    0,
    0,
    scale,
    0,
    0,
    0,
    1,
  ]);

  // final transform to panorama pixels: P = S * T * H_i * src
  const finalH: Float32Array[] = (homos as Float32Array[]).map((h, idx) => {
    const fh = matMul3(S, matMul3(T, h));
    debug(`stitch: finalH[${idx}]`, Array.from(fh));
    return fh;
  });

  // Build per-image seam masks (destination-space)
  const seamMasks = await buildSeamMasks(
    cylBitmaps,
    finalH,
    dstW,
    dstH,
    feather,
    seamWidthPx
  );
  // Fallback single-pixel white mask for images without a seam mask
  const whiteMaskBmp = await (async () => {
    const mc = document.createElement("canvas");
    mc.width = 1;
    mc.height = 1;
    const mctx = mc.getContext("2d")!;
    mctx.fillStyle = "#fff";
    mctx.fillRect(0, 0, 1, 1);
    return await createImageBitmap(mc);
  })();

  setProgress?.(52);

  // 3) GPU setup: target texture & pipeline
  const colorTex = device.createTexture({
    size: { width: dstW, height: dstH },
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const sampler = device.createSampler({
    minFilter: "linear",
    magFilter: "linear",
  });

  const module = device.createShaderModule({ code: warpWGSL });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs_main" },
    fragment: {
      module,
      entryPoint: "fs_main",
      targets: [
        {
          format: "rgba8unorm",
          blend: {
            color: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: { topology: "triangle-list" },
  });

  // 4) Upload source textures and draw each with its inverse homography
  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: colorTex.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  pass.setPipeline(pipeline);

  for (let i = 0; i < cylBitmaps.length; i++) {
    const b = cylBitmaps[i];
    debug(`stitch: drawing image ${i}`, { w: b.width, h: b.height });
    const tex = createTextureFromBitmap(device, b);

    const maskBmp = seamMasks[i] ?? whiteMaskBmp;
    const maskTex = createTextureFromBitmap(device, maskBmp);

    const invH = matInvert3(finalH[i]); // map dest->src

    // WGSL/std140-style packing:
    // mat3x3<f32> is 3 columns of vec3<f32>, each with 16-byte stride (pad one float per column)
    // Then: vec2 srcSize, vec2 dstSize, f32 feather, f32 _pad, plus struct tail padding to 16-byte multiple.
    const uniformData = new Float32Array(20); // 20 * 4 = 80 bytes (matches minBindingSize)

    // invH columns (column-major) with padding at the 4th element of each column
    uniformData[0] = invH[0]; // col0.x
    uniformData[1] = invH[1]; // col0.y
    uniformData[2] = invH[2]; // col0.z
    uniformData[3] = 0.0; // pad

    uniformData[4] = invH[3]; // col1.x
    uniformData[5] = invH[4]; // col1.y
    uniformData[6] = invH[5]; // col1.z
    uniformData[7] = 0.0; // pad

    uniformData[8] = invH[6]; // col2.x
    uniformData[9] = invH[7]; // col2.y
    uniformData[10] = invH[8]; // col2.z
    uniformData[11] = 0.0; // pad

    // srcSize (vec2), dstSize (vec2)
    uniformData[12] = b.width;
    uniformData[13] = b.height;
    uniformData[14] = dstW;
    uniformData[15] = dstH;

    // feather + explicit pad, then tail padding to hit 80 bytes
    uniformData[16] = feather;
    uniformData[17] = 0.0; // _pad
    uniformData[18] = 0.0; // tail pad
    uniformData[19] = 0.0; // tail pad

    const ubuf = device.createBuffer({
      size: uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(ubuf, 0, uniformData);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: tex.createView() },
        { binding: 2, resource: { buffer: ubuf } },
        { binding: 3, resource: maskTex.createView() },
      ],
    });

    pass.setBindGroup(0, bindGroup);
    pass.draw(3, 1, 0, 0);

    const val = 52 + Math.floor(((i + 1) / cylBitmaps.length) * 38);
    debug("progress:", val);
    setProgress?.(val);
  }

  pass.end();
  device.queue.submit([commandEncoder.finish()]);

  // 5) Read back and convert to data URL
  debug("stitch: readback");
  const imageData = await readTextureToImageData(device, colorTex, dstW, dstH);
  const c = document.createElement("canvas");
  c.width = dstW;
  c.height = dstH;
  const ctx = c.getContext("2d")!;
  ctx.putImageData(imageData, 0, 0);

  setProgress?.(98);
  debug("progress:", 98);
  const dataUrl = c.toDataURL("image/png");
  setProgress?.(100);
  debug("progress:", 100);
  return dataUrl;
}

// Load files into downscaled ImageBitmaps
async function filesToBitmaps(
  files: File[],
  maxDim: number,
  onStep?: (idx: number, total: number) => void
): Promise<ImageBitmap[]> {
  debug("filesToBitmaps: count", files.length, "maxDim", maxDim);
  const out: ImageBitmap[] = [];
  for (let i = 0; i < files.length; i++) {
    onStep?.(i, files.length);
    const f = files[i];
    debug(`file[${i}]`, { name: f.name, size: f.size, type: f.type });
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.src = url;
    await img.decode();
    debug(`file[${i}] decoded`, { w: img.naturalWidth, h: img.naturalHeight });
    const bmp = await createImageBitmap(img);
    const ds = await downscaleBitmap(bmp, maxDim);
    debug(`file[${i}] downscaled`, { w: ds.width, h: ds.height });
    out.push(ds);
    URL.revokeObjectURL(url);
  }
  onStep?.(files.length, files.length);
  return out;
}

// --------------- React Component ---------------
export default function WebGPUPanorama() {
  const [seamWidth, setSeamWidth] = useState<number>(48); // pixels in final panorama
  const [files, setFiles] = useState<File[]>([]);
  const [bitmaps, setBitmaps] = useState<ImageBitmap[] | null>(null);
  const [mode, setMode] = useState<
    "idle" | "confirm" | "stitching" | "preview"
  >("idle");
  const [progress, setProgress] = useState(0);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [maxDim, setMaxDim] = useState<number>(1600);
  const [feather, setFeather] = useState<number>(60);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const onSelectFiles = useCallback(
    async (incoming: FileList | null) => {
      setError(null);
      if (!incoming || incoming.length === 0) return;
      const list = Array.from(incoming).filter((f) =>
        f.type.startsWith("image/")
      );
      if (list.length < 2) {
        setError("Please select at least two images to stitch.");
        return;
      }
      setFiles(list);

      setProgress(5);
      const bms = await filesToBitmaps(list, maxDim, (i, total) =>
        setProgress(Math.floor((i / total) * 20))
      );
      setBitmaps(bms);
      debug(
        "onSelectFiles: bitmaps prepared",
        bms.map((b) => [b.width, b.height])
      );
      setMode("confirm");
    },
    [maxDim]
  );

  const startOver = useCallback(() => {
    setMode("idle");
    setProgress(0);
    setResultUrl(null);
    setError(null);
    setBitmaps(null);
    setFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const startStitch = useCallback(async () => {
    if (!bitmaps || bitmaps.length < 2) return;
    setMode("stitching");
    setProgress(0);
    setError(null);
    try {
      debug("startStitch: begin");
      const url = await stitchWithWebGPU(bitmaps, feather, seamWidth, (p) =>
        setProgress(p)
      );
      setResultUrl(url);
      setMode("preview");
    } catch (e: any) {
      console.error(e);
      debug("startStitch: error", e);
      setError(e?.message || "Failed to stitch – see console for details.");
      // Try naive fallback
      try {
        const url = await naiveStitch(bitmaps);
        setResultUrl(url);
        setMode("preview");
      } catch (e2) {
        console.error("Fallback failed:", e2);
        setMode("confirm");
      }
    }
  }, [bitmaps, feather]);

  const download = useCallback(() => {
    if (!resultUrl) return;
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = "panorama.png";
    a.click();
  }, [resultUrl]);

  const thumbnails = useMemo(() => {
    if (!files.length) return null;
    return files.map((f, i) => (
      <div
        key={i}
        className="w-28 h-20 rounded-xl overflow-hidden bg-muted flex items-center justify-center text-xs"
      >
        <img
          src={URL.createObjectURL(f)}
          className="object-cover w-full h-full"
          onLoad={(e) =>
            URL.revokeObjectURL((e.target as HTMLImageElement).src)
          }
        />
      </div>
    ));
  }, [files]);

  return (
    <div className="mx-auto max-w-5xl p-6">
      <div className="mb-6 flex items-center gap-3">
        <Layers className="w-6 h-6 text-primary" />
        <h1 className="text-2xl font-semibold">WebGPU Panorama Stitcher</h1>
      </div>

      <AnimatePresence mode="wait">
        {mode === "idle" && (
          <motion.div
            key="idle"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            <Card className="border-dashed">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="w-5 h-5" /> Select Images
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  <div className="flex items-center gap-4">
                    <Input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={(e) => onSelectFiles(e.target.files)}
                    />
                    <Button
                      variant="secondary"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <ImagePlus className="w-4 h-4 mr-2" /> Choose Files
                    </Button>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="maxdim">Max side (px)</Label>
                      <Input
                        id="maxdim"
                        type="number"
                        value={maxDim}
                        onChange={(e) =>
                          setMaxDim(Number(e.target.value || 1600))
                        }
                      />
                    </div>
                    <div>
                      <Label htmlFor="feather">Feather (px)</Label>
                      <Input
                        id="feather"
                        type="number"
                        value={feather}
                        onChange={(e) =>
                          setFeather(Number(e.target.value || 60))
                        }
                      />
                    </div>
                    <div className="text-sm text-muted-foreground flex items-end">
                      Images are downscaled for performance. Increase only if
                      you have a beefy GPU.
                    </div>
                    <div className="flex items-center gap-2">
                      <Label htmlFor="seamWidth">Seam width (px)</Label>
                      <Input
                        id="seamWidth"
                        type="number"
                        min={0}
                        max={800}
                        className="w-24"
                        value={seamWidth}
                        onChange={(e) =>
                          setSeamWidth(Math.max(0, Number(e.target.value) || 0))
                        }
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {mode === "confirm" && (
          <motion.div
            key="confirm"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-4"
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5" /> Confirm Selection
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
                  {thumbnails}
                </div>
                <div className="flex items-center justify-between pt-2">
                  <div className="text-sm text-muted-foreground">
                    {files.length} images selected
                  </div>
                  <div className="flex gap-2">
                    <Button variant="ghost" onClick={startOver}>
                      <ArrowLeft className="w-4 h-4 mr-2" /> Start over
                    </Button>
                    <Button onClick={startStitch}>
                      Start stitching <ChevronRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {mode === "stitching" && (
          <motion.div
            key="stitching"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="w-5 h-5" /> Stitching…
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <div className="text-sm text-muted-foreground">
                      This may take a minute for large inputs.
                    </div>
                  </div>
                  <Progress value={progress} />
                  <div className="text-xs text-muted-foreground">
                    {progress}%
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {mode === "preview" && (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-4"
          >
            <Card>
              <CardHeader>
                <CardTitle>Preview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {resultUrl ? (
                  <img
                    src={resultUrl}
                    alt="Panorama result"
                    className="w-full h-auto rounded-xl shadow"
                  />
                ) : (
                  <div className="text-sm text-muted-foreground">
                    No result to show.
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <Button variant="ghost" onClick={() => setMode("confirm")}>
                    <ArrowLeft className="w-4 h-4 mr-2" /> Go back
                  </Button>
                  <Button onClick={download}>
                    <Download className="w-4 h-4 mr-2" /> Download
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <div className="mt-4 text-sm text-destructive flex items-center gap-2">
          <TriangleAlert className="w-4 h-4" /> {error}
        </div>
      )}

      <div className="mt-6 text-xs text-muted-foreground">
        Tip: For best results, supply overlapping images shot from a fixed
        rotation (handheld sweep). If WebGPU or OpenCV isn’t available, the app
        will fall back to a simple paste.
      </div>
    </div>
  );
}
