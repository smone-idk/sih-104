import { chromium } from "playwright";
const SHOT = "/tmp/claude-1000/-home-arpan-projects/cd7ec959-7cb6-4e47-8f8e-9581af7b4fb3/scratchpad";
const b = await chromium.launch();
// NOTE: no microphone permission granted — proves nothing tries to open it.
const p = await b.newPage({ viewport: { width: 1500, height: 1300 } });
const errs = [];
const mediaCalls = [];
p.on("pageerror", e => errs.push("pageerror: " + e.message));
p.on("console", m => { if (m.type() === "error") errs.push("console: " + m.text()); });

// Instrument getUserMedia so ANY call is recorded, from anywhere in the app.
await p.addInitScript(() => {
  window.__gum = [];
  const md = navigator.mediaDevices;
  if (md && md.getUserMedia) {
    const orig = md.getUserMedia.bind(md);
    md.getUserMedia = (...a) => { window.__gum.push(Date.now()); return orig(...a); };
  }
});

for (const path of ["/live", "/upload", "/profiles", "/approvals", "/incidents", "/system"]) {
  await p.goto("http://localhost:5199" + path, { waitUntil: "networkidle" });
  await p.waitForTimeout(700);
  const n = await p.evaluate(() => (window.__gum || []).length);
  mediaCalls.push(`${path}: ${n}`);
}
console.log("getUserMedia calls per page (must all be 0):", mediaCalls.join("  "));

await p.goto("http://localhost:5199/profiles", { waitUntil: "networkidle" });
await p.waitForTimeout(900);
const rows = await p.locator("tbody tr").count();
const banner = await p.locator("text=No enrolled voice profile").count();
console.log(`profiles listed: ${rows}, "no profile" banner: ${banner}`);
await p.screenshot({ path: `${SHOT}/p5-profiles.png`, fullPage: true });

// recording indicator must not be present before pressing
const indBefore = await p.locator("text=RECORDING").count();
console.log("RECORDING indicator before press:", indBefore);

await p.goto("http://localhost:5199/upload", { waitUntil: "networkidle" });
await p.waitForTimeout(600);
await p.screenshot({ path: `${SHOT}/p5-upload.png`, fullPage: true });
const gumFinal = await p.evaluate(() => (window.__gum || []).length);
console.log("getUserMedia calls after visiting Upload (mic UI rendered):", gumFinal);
console.log(errs.length ? "JS ERRORS:\n" + errs.join("\n") : "no JS errors");
await b.close();
