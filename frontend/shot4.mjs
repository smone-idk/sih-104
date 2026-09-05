import { chromium } from "playwright";
const SHOT = "/tmp/claude-1000/-home-arpan-projects/cd7ec959-7cb6-4e47-8f8e-9581af7b4fb3/scratchpad";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1500, height: 1500 } });
const errs = [];
p.on("pageerror", e => errs.push("pageerror: " + e.message));
p.on("console", m => { if (m.type() === "error") errs.push("console: " + m.text()); });

await p.goto("http://localhost:5199/approvals", { waitUntil: "networkidle" });
await p.waitForTimeout(1200);
const approveBtn = p.locator('button:has-text("Approve ₹")');
console.log("Approve button disabled:", await approveBtn.isDisabled());
await p.click('button:has-text("Try to bypass")');
await p.waitForSelector('text=HTTP 403', { timeout: 20000 });
await p.waitForTimeout(400);
await p.screenshot({ path: `${SHOT}/p4-approvals.png`, fullPage: true });
console.log("forged bypass ->", await p.locator('span:has-text("HTTP 403")').first().textContent());

await p.goto("http://localhost:5199/incidents", { waitUntil: "networkidle" });
await p.waitForTimeout(800);
const rows = await p.locator("tbody tr").count();
console.log("incident rows:", rows);
await p.screenshot({ path: `${SHOT}/p4-incidents.png`, fullPage: true });
console.log(errs.length ? "JS ERRORS:\n" + errs.join("\n") : "no JS errors");
await b.close();
