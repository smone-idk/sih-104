import { chromium } from "playwright";
const SHOT = "/tmp/claude-1000/-home-arpan-projects/cd7ec959-7cb6-4e47-8f8e-9581af7b4fb3/scratchpad";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1500, height: 1400 } });
const errs = [];
p.on("pageerror", e => errs.push(e.message));

async function speakerRow() {
  await p.goto("http://localhost:5199/live", { waitUntil: "networkidle" });
  await p.selectOption("select", "genuine_control");
  const rt = p.locator('input[type=checkbox]').nth(2);
  if (await rt.isChecked()) await rt.uncheck();
  await p.click('button:has-text("Start simulation")');
  await p.waitForSelector('h2:has-text("Explainability")', { timeout: 180000 });
  await p.waitForTimeout(400);
  const row = p.locator('tr', { has: p.locator('text=Speaker consistency') }).first();
  return (await row.innerText()).replace(/\s+/g, " ").trim();
}

console.log("WITH profile   :", await speakerRow());

await p.goto("http://localhost:5199/profiles", { waitUntil: "networkidle" });
await p.waitForTimeout(600);
await p.click('button:has-text("Delete")');
await p.waitForSelector("text=No enrolled voice profile", { timeout: 20000 });
await p.waitForTimeout(400);
await p.screenshot({ path: `${SHOT}/p5-profile-deleted.png`, fullPage: true });
console.log("banner after delete: shown");

console.log("WITHOUT profile:", await speakerRow());
await p.screenshot({ path: `${SHOT}/p5-gate-live.png`, fullPage: true });
console.log(errs.length ? "JS ERRORS: " + errs.join("; ") : "no JS errors");
await b.close();
