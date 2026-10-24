// JSON-line v1 protocol. Gameplay uses the real SDK; evidence comes from the engine.
import { BotSDK } from '../rs-sdk/sdk/index';
import { BotActions } from '../rs-sdk/sdk/actions';
import puppeteer from 'puppeteer';
import { createInterface } from 'node:readline';

console.log = console.info = console.warn = (..._args: unknown[]) => {}; // SDK may log credential URLs.
const episode = process.env.RLVR_EPISODE_ID!;
const world = process.env.RLVR_WORLD_ID!;
const control = process.env.RLVR_CONTROL_URL!;
const token = process.env.RLVR_CONTROL_TOKEN!;
const sdk = new BotSDK({ botUsername: process.env.RS_BOT_USERNAME!, password: process.env.RS_BOT_PASSWORD!,
    gatewayUrl: process.env.RS_GATEWAY!, autoLaunchBrowser: false, autoReconnect: false, showChat: false });
const bot = new BotActions(sdk);
let browser: Awaited<ReturnType<typeof puppeteer.launch>> | undefined;
let poisoned = false;
let lastId = 0;
const unavailableUntil = new Map<string, number>();
const resourceKey = (r: any) => `${r.id}:${r.x}:${r.z}`;
class ActionBudgetReached extends Error {}

function diagnose(error: unknown): void {
    let message = String(error);
    for (const secret of [token, process.env.RS_BOT_PASSWORD]) {
        if (secret) message = message.split(secret).join('[redacted]');
    }
    process.stderr.write(message.replace(/https?:\/\/\S+/g, '[url]') + '\n');
}

async function engine(path: string, method = 'GET'): Promise<any> {
    const response = await fetch(`${control}/${path}`, { method,
        headers: { authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error(`Engine control ${path} failed (${response.status})`);
    return await response.json();
}

async function until(predicate: (s: any) => boolean, deadline: number): Promise<any> {
    while (Date.now() < deadline) {
        const state = await engine('state');
        if (predicate(state)) return state;
        await Bun.sleep(50);
    }
    throw new ActionBudgetReached('Action observation budget exhausted');
}
const count = (s: any, id: number) => s.inventory.reduce((n: number, i: any) => n + (i.id === id ? i.count : 0), 0);

async function action(name: string, args: any, deadline: number): Promise<any> {
    const start = performance.now();
    const before = await engine('state');
    if (before.pending) throw new Error('Uncancelled activity before action');
    let status = 'failed';
    let reason = 'precondition';
    let target: any = null;
    if (name === 'chopTree') {
        if (args.tree_type && args.tree_type !== 'Tree') throw new Error('Unsupported resource type');
        target = before.resources.filter((r: any) => (unavailableUntil.get(resourceKey(r)) ?? 0) < Date.now()).sort((a: any, b: any) =>
            Math.abs(a.x - before.position[0]) + Math.abs(a.z - before.position[1]) -
            Math.abs(b.x - before.position[0]) - Math.abs(b.z - before.position[1]))[0];
        if (target && before.inventory.length < 28 &&
            [...before.inventory, ...before.equipment].some(i => i.id === 1351)) {
            const result = await sdk.sendInteractLoc(target.x, target.z, target.id, 1);
            diagnose(`action_accepted:${result.success}`);
            if (result.success) {
                try {
                const observed = await until(s => s.tick > before.tick &&
                    (count(s, 1511) > count(before, 1511) || !s.resources.some((r: any) =>
                        r.id === target.id && r.x === target.x && r.z === target.z)), deadline - 2000);
                status = count(observed, 1511) > count(before, 1511) ? 'completed' : 'failed';
                reason = status === 'completed' ? 'harvested' : 'resource_unavailable';
                diagnose(`action_observed:${reason}`);
                } catch (error) {
                    if (!(error instanceof ActionBudgetReached)) throw error;
                    diagnose(`action_budget:${JSON.stringify(await engine('state'))}`);
                    reason = 'no_harvest_within_action_budget';
                    unavailableUntil.set(resourceKey(target), Date.now() + 60000);
                }
            } else reason = 'interaction_rejected';
        }
    } else if (name === 'walkTo') {
        if (!Number.isInteger(args.x) || !Number.isInteger(args.z)) throw new Error('Invalid coordinates');
        const result = await bot.walkTo(args.x, args.z);
        const observed = await engine('state');
        status = result.success && observed.position[0] === args.x && observed.position[1] === args.z ? 'completed' : 'failed';
        reason = status === 'completed' ? 'arrived' : 'path_failed';
    } else if (name === 'equipItem') {
        if (args.item !== 'Bronze axe') throw new Error('Only the fixture axe can be equipped');
        const result = await bot.equipItem(/^Bronze axe$/i);
        const observed = await engine('state');
        status = result.success && observed.equipment.some((i: any) => i.id === 1351) ? 'completed' : 'failed';
        reason = status === 'completed' ? 'equipped' : 'equip_failed';
    } else throw new Error('Unsupported action');
    const canceled = await engine('cancel', 'POST');
    let after = canceled;
    while (Date.now() < deadline) {
        after = await engine('state');
        if (after.tick > canceled.tick && !after.pending) break;
        if (after.pending) await engine('cancel', 'POST');
        await Bun.sleep(50);
    }
    if (after.tick <= canceled.tick || after.pending) throw new Error('Cancellation outcome unknown');
    // A harvest may land on the cancellation tick; the final engine state decides.
    if (name === 'chopTree' && count(after, 1511) > count(before, 1511)) {
        status = 'completed'; reason = 'harvested';
    }
    return { name, status, reason, before, after, target, elapsed_seconds: (performance.now() - start) / 1000 };
}

function send(value: object): void { process.stdout.write(JSON.stringify(value) + '\n'); }
async function close(): Promise<void> {
    sdk.disconnect();
    if (browser) {
        const timer = setTimeout(() => browser?.process()?.kill('SIGKILL'), 2000);
        try { await browser.close(); } finally { clearTimeout(timer); }
    }
}
process.on('SIGTERM', () => { void close().finally(() => process.exit(0)); });
process.on('SIGINT', () => { void close().finally(() => process.exit(0)); });

try {
    diagnose('stage:launch_browser');
    browser = await puppeteer.launch({ headless: true, args: ['--disable-dev-shm-usage', '--use-gl=angle'] });
    diagnose('stage:browser_launched');
    const page = await browser.newPage();
    page.on('pageerror', diagnose);
    page.on('console', message => diagnose(message.text()));
    page.on('response', response => { if (response.status() >= 400) diagnose(`HTTP ${response.status()} ${response.url()}`); });
    const url = new URL('/bot', process.env.RLVR_WEB_URL);
    url.searchParams.set('bot', process.env.RS_BOT_USERNAME!);
    url.searchParams.set('password', process.env.RS_BOT_PASSWORD!);
    await page.goto(url.toString(), { waitUntil: 'domcontentloaded', timeout: 60000 });
    diagnose(`stage:page_loaded:${await page.title()}`);
    await sdk.connect();
    diagnose('stage:sdk_connected');
    const end = Date.now() + 60000;
    while (!sdk.getState()?.inGame && Date.now() < end) await Bun.sleep(100);
    if (!sdk.getState()?.inGame) throw new Error('Browser did not enter game');
    await sdk.waitForTicks(3);
    const state = await engine('bootstrap', 'POST');
    send({ version: 1, episode_id: episode, world_id: world, request_id: '0', status: 'ready', state });
    for await (const line of createInterface({ input: process.stdin })) {
        let cmd: any;
        try {
            cmd = JSON.parse(line);
            if (poisoned || cmd.version !== 1 || cmd.episode_id !== episode || cmd.world_id !== world ||
                !Number.isInteger(Number(cmd.request_id)) || Number(cmd.request_id) !== ++lastId ||
                !Number.isFinite(cmd.timeout_ms) || cmd.timeout_ms <= 0 || cmd.timeout_ms > 30000) {
                throw new Error('Invalid request envelope');
            }
            let timer: ReturnType<typeof setTimeout>;
            const deadline = Date.now() + cmd.timeout_ms;
            const task = cmd.operation === 'state' ? engine('state') :
                cmd.operation === 'action' ? action(cmd.name, cmd.arguments, deadline) :
                Promise.reject(new Error('Unknown operation'));
            const result = await Promise.race([task, new Promise((_, reject) => {
                timer = setTimeout(() => reject(new Error('Request deadline exceeded')), cmd.timeout_ms);
            })]).finally(() => clearTimeout(timer!));
            send({ version: 1, request_id: cmd.request_id, episode_id: episode, world_id: world, status: 'ok', result });
        } catch (_error) {
            diagnose(_error);
            poisoned = true;
            send({ version: 1, request_id: cmd?.request_id ?? '', episode_id: episode, world_id: world,
                status: 'infra_error', reason: 'bridge_request_failed_or_unknown' });
            break;
        }
    }
} catch (_error) {
    diagnose(_error);
    send({ version: 1, episode_id: episode, world_id: world, request_id: '0', status: 'infra_error', reason: 'startup_failed' });
    process.exitCode = 1;
} finally { await close(); }
