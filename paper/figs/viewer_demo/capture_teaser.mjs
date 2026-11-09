// High-resolution screenshots of the viewer for the teaser (Fig. 1), via headless Chrome and the DevTools protocol.
// Needs the viewer running on Gemma-3-4B-it at http://localhost:8768 (launch config "flow-gemma4b-it").
//     node paper/figs/viewer_demo/capture_teaser.mjs
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "teaser");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9333, W = 1500, H = 1720, SCALE = 3;
const PROMPT = "Fact: the capital of the state containing Dallas is";

const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, "--no-first-run",
  `--user-data-dir=${mkdtempSync(join(tmpdir(), "teaser-"))}`, "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()).find(t => t.type === "page"); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener("open", r));
let id = 0; const pending = new Map();
ws.addEventListener("message", e => { const m = JSON.parse(e.data); if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const js = async expr => {
  const m = await send("Runtime.evaluate", { expression: `(async () => { ${expr} })()`, awaitPromise: true, returnByValue: true });
  if (m.result.exceptionDetails) throw new Error(JSON.stringify(m.result.exceptionDetails));
  return m.result.result.value;
};
const shot = async (name, [x0, y0, x1, y1]) => {
  const m = await send("Page.captureScreenshot", { format: "png", clip: { x: x0, y: y0, width: x1 - x0, height: y1 - y0, scale: 1 } });
  writeFileSync(join(OUT, name), Buffer.from(m.result.data, "base64"));
  console.log("wrote", name);
};
const waitIdle = `await new Promise(r => setTimeout(r, 400)); for (let i = 0; i < 600 && /ing|…/.test(document.getElementById("status").innerText); i++) await new Promise(r => setTimeout(r, 200));`;
const rect = sel => js(`const b = document.querySelector(${JSON.stringify(sel)}).getBoundingClientRect(); return [b.left, b.top, b.right, b.bottom];`);

await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: SCALE, mobile: false });
await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
await send("Page.navigate", { url: "http://localhost:8768/" });
await sleep(3000);
await js(waitIdle);

// Reading: the Dallas question with Texas highlighted, and the Dallas cell at layer 21 selected.
await js(`const p = document.getElementById("prompt"); p.value = ${JSON.stringify(PROMPT)}; p.dispatchEvent(new Event("input", {bubbles: true}));
          document.getElementById("run").click(); ${waitIdle}`);
await js(`const t = document.getElementById("trackword"); t.value = "Texas";
          t.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter", bubbles: true})); ${waitIdle}`);
await js(`document.querySelector('#gridbox g.cell[data-l="21"][data-p="9"]').dispatchEvent(new MouseEvent("click", {bubbles: true}));
          await new Promise(r => setTimeout(r, 1500));`);
// For the paper, label highlighted cells with their top word, as in the unhighlighted grid, instead of Texas's rank
// (the shading still shows the rank). The top word is the first line of each cell's tooltip.
await js(`for (const c of document.querySelectorAll("#gridbox g.cell")) {
            const t = c.querySelector("text"); if (!t || !t.textContent.startsWith("#")) continue;
            const m = c.querySelector("title").textContent.split("\\n")[1].match(/^'(.*)' [\\d.]+%$/);
            if (m) t.textContent = m[1].length > 7 ? m[1].slice(0, 6) + "…" : m[1];
          }`);
await shot("a_reading_grid.png", await rect("#gridbox"));
await shot("a_reading_rail.png", await rect("#rail"));

// Editing: swap Texas for California along J-lens directions, strength 4; "What changed" shading.
await js(`document.getElementById("editfold").open = true;
          const k = document.getElementById("ekind"); k.value = "swap"; k.dispatchEvent(new Event("change", {bubbles: true}));
          await new Promise(r => setTimeout(r, 300));
          const set = (i, v) => { const e = document.getElementById(i); e.value = v; e.dispatchEvent(new Event("input", {bubbles: true})); e.dispatchEvent(new Event("change", {bubbles: true})); };
          set("eo", "Texas"); set("en", "California"); set("ess", "4"); set("e-sdir", "jlens");
          document.getElementById("ego").click();
          for (let i = 0; i < 600 && !document.getElementById("editbar").getBoundingClientRect().height; i++) await new Promise(r => setTimeout(r, 200));
          await new Promise(r => setTimeout(r, 1000));`);
await js(`document.getElementById("trackclear").click(); await new Promise(r => setTimeout(r, 1000));`);
console.log(await js(`return document.getElementById("output").innerText`));
await shot("b_edit_bar.png", await rect("#editbar"));
await shot("b_edit_grid.png", await rect("#gridbox"));
await shot("b_edit_rail.png", await rect("#rail"));
writeFileSync(join(OUT, "cells.json"), JSON.stringify(await js(`
  const g = document.getElementById("gridbox").getBoundingClientRect();
  return [...document.querySelectorAll("#gridbox g.cell")].map(c => { const b = c.getBoundingClientRect();
    return {l: +c.dataset.l, p: +c.dataset.p, x0: b.left - g.left, y0: b.top - g.top, x1: b.right - g.left, y1: b.bottom - g.top}; });`)));
ws.close(); chrome.kill();
