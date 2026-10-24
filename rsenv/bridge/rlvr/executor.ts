// JSON-line v1 protocol. Gameplay uses the real SDK; evidence comes from the engine.
import { BotSDK } from '../rs-sdk/sdk/index';
import { BotActions } from '../rs-sdk/sdk/actions';
import puppeteer from 'puppeteer';
import { createInterface } from 'node:readline';
import type { Observation, Resource } from './engine_control';

console.log = console.info = console.warn = (..._args: unknown[]) => {}; // SDK may log credential URLs.
const episode = process.env.RLVR_EPISODE_ID!;
const world = process.env.RLVR_WORLD_ID!;
const control = process.env.RLVR_CONTROL_URL!;
const token = process.env.RLVR_CONTROL_TOKEN!;
const sdk = new BotSDK({
    botUsername: process.env.RS_BOT_USERNAME!,
    password: process.env.RS_BOT_PASSWORD!,
    gatewayUrl: process.env.RS_GATEWAY!,
    autoLaunchBrowser: false,
    autoReconnect: false,
    showChat: false,
});
const bot = new BotActions(sdk);
let browser: Awaited<ReturnType<typeof puppeteer.launch>> | undefined;
let lastId = 0;
const unavailableUntil = new Map<string, number>();
const resourceKey = (r: Resource): string => `${r.id}:${r.x}:${r.z}`;
class ActionBudgetReached extends Error {}

function diagnose(error: unknown): void {
    let message = String(error);
    for (const secret of [token, process.env.RS_BOT_PASSWORD]) {
        if (secret) message = message.split(secret).join('[redacted]');
    }
    process.stderr.write(message.replace(/https?:\/\/\S+/g, '[url]') + '\n');
}

async function engine(path: string, method = 'GET'): Promise<Observation> {
    const response = await fetch(`${control}/${path}`, {
        method,
        headers: { authorization: `Bearer ${token}` },
        signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error(`Engine control ${path} failed (${response.status})`);
    return await response.json();
}

async function until(predicate: (s: Observation) => boolean, deadline: number): Promise<Observation> {
    while (Date.now() < deadline) {
        const state = await engine('state');
        if (predicate(state)) return state;
        await Bun.sleep(50);
    }
    throw new ActionBudgetReached('Action observation budget exhausted');
}
const count = (s: Observation, id: number): number => s.inventory.reduce((n, i) => n + (i.id === id ? i.count : 0), 0);

interface ActionOutcome {
    name: string;
    status: 'completed' | 'failed';
    reason: string;
    before: Observation;
    after: Observation;
    target: Resource | null;
    elapsed_seconds: number;
}

interface Command {
    version: number;
    episode_id: string;
    world_id: string;
    request_id: string;
    timeout_ms: number;
    operation: string;
    name: string;
    arguments: Record<string, unknown>;
}

async function action(name: string, args: Record<string, unknown>, deadline: number): Promise<ActionOutcome> {
    const start = performance.now();
    const before = await engine('state');
    if (before.pending) throw new Error('Uncancelled activity before action');
    let status: ActionOutcome['status'] = 'failed';
    let reason = 'precondition';
    let target: Resource | null = null;
    if (name === 'chopTree') {
        if (args.tree_type && args.tree_type !== 'Tree') throw new Error('Unsupported resource type');
        const tree = before.resources
            .filter((r) => (unavailableUntil.get(resourceKey(r)) ?? 0) < Date.now())
            .sort(
                (a, b) =>
                    Math.abs(a.x - before.position[0]) +
                    Math.abs(a.z - before.position[1]) -
                    Math.abs(b.x - before.position[0]) -
                    Math.abs(b.z - before.position[1]),
            )[0];
        target = tree ?? null;
        if (
            tree &&
            before.inventory.length < 28 &&
            [...before.inventory, ...before.equipment].some((i) => i.id === 1351)
        ) {
            const result = await sdk.sendInteractLoc(tree.x, tree.z, tree.id, 1);
            diagnose(`action_accepted:${result.success}`);
            if (result.success) {
                try {
                    const observed = await until(
                        (s) =>
                            s.tick > before.tick &&
                            (count(s, 1511) > count(before, 1511) ||
                                !s.resources.some((r) => r.id === tree.id && r.x === tree.x && r.z === tree.z)),
                        deadline - 2000,
                    );
                    status = count(observed, 1511) > count(before, 1511) ? 'completed' : 'failed';
                    reason = status === 'completed' ? 'harvested' : 'resource_unavailable';
                    diagnose(`action_observed:${reason}`);
                } catch (error) {
                    if (!(error instanceof ActionBudgetReached)) throw error;
                    diagnose(`action_budget:${JSON.stringify(await engine('state'))}`);
                    reason = 'no_harvest_within_action_budget';
                    unavailableUntil.set(resourceKey(tree), Date.now() + 60000);
                }
            } else reason = 'interaction_rejected';
        }
    } else if (name === 'walkTo') {
        if (
            typeof args.x !== 'number' ||
            typeof args.z !== 'number' ||
            !Number.isInteger(args.x) ||
            !Number.isInteger(args.z)
        )
            throw new Error('Invalid coordinates');
        const result = await bot.walkTo(args.x, args.z);
        const observed = await engine('state');
        status =
            result.success && observed.position[0] === args.x && observed.position[1] === args.z
                ? 'completed'
                : 'failed';
        reason = status === 'completed' ? 'arrived' : 'path_failed';
    } else if (name === 'equipItem') {
        if (args.item !== 'Bronze axe') throw new Error('Only the fixture axe can be equipped');
        const result = await bot.equipItem(/^Bronze axe$/i);
        const observed = await engine('state');
        status = result.success && observed.equipment.some((i) => i.id === 1351) ? 'completed' : 'failed';
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
        status = 'completed';
        reason = 'harvested';
    }
    return { name, status, reason, before, after, target, elapsed_seconds: (performance.now() - start) / 1000 };
}

function send(value: object): void {
    process.stdout.write(JSON.stringify(value) + '\n');
}
async function close(): Promise<void> {
    sdk.disconnect();
    if (browser) {
        const timer = setTimeout(() => browser?.process()?.kill('SIGKILL'), 2000);
        try {
            await browser.close();
        } finally {
            clearTimeout(timer);
        }
    }
}
process.on('SIGTERM', () => {
    void close().finally(() => process.exit(0));
});
process.on('SIGINT', () => {
    void close().finally(() => process.exit(0));
});

try {
    diagnose('stage:launch_browser');
    browser = await puppeteer.launch({ headless: true, args: ['--disable-dev-shm-usage', '--use-gl=angle'] });
    diagnose('stage:browser_launched');
    const page = await browser.newPage();
    page.on('pageerror', diagnose);
    page.on('console', (message) => diagnose(message.text()));
    page.on('response', (response) => {
        if (response.status() >= 400) diagnose(`HTTP ${response.status()} ${response.url()}`);
    });
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
        let requestId = '';
        try {
            const cmd: Command = JSON.parse(line);
            requestId = cmd.request_id;
            if (
                cmd.version !== 1 ||
                cmd.episode_id !== episode ||
                cmd.world_id !== world ||
                !Number.isInteger(Number(cmd.request_id)) ||
                Number(cmd.request_id) !== lastId + 1 ||
                !Number.isFinite(cmd.timeout_ms) ||
                cmd.timeout_ms <= 0 ||
                cmd.timeout_ms > 30000
            ) {
                throw new Error('Invalid request envelope');
            }
            lastId += 1;
            let timer: ReturnType<typeof setTimeout>;
            const deadline = Date.now() + cmd.timeout_ms;
            let task: Promise<Observation | ActionOutcome>;
            if (cmd.operation === 'state') {
                task = engine('state');
            } else if (cmd.operation === 'action') {
                task = action(cmd.name, cmd.arguments, deadline);
            } else {
                throw new Error('Unknown operation');
            }
            const result = await Promise.race([
                task,
                new Promise((_, reject) => {
                    timer = setTimeout(() => reject(new Error('Request deadline exceeded')), cmd.timeout_ms);
                }),
            ]).finally(() => clearTimeout(timer!));
            send({
                version: 1,
                request_id: cmd.request_id,
                episode_id: episode,
                world_id: world,
                status: 'ok',
                result,
            });
        } catch (_error) {
            diagnose(_error);
            send({
                version: 1,
                request_id: requestId,
                episode_id: episode,
                world_id: world,
                status: 'infra_error',
                reason: 'bridge_request_failed_or_unknown',
            });
            break;
        }
    }
} catch (_error) {
    diagnose(_error);
    send({
        version: 1,
        episode_id: episode,
        world_id: world,
        request_id: '0',
        status: 'infra_error',
        reason: 'startup_failed',
    });
    process.exitCode = 1;
} finally {
    await close();
}
