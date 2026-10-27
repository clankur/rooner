// JSON-line v2 protocol. Gameplay uses the real SDK; evidence comes from the engine.
import { BotSDK } from '../rs-sdk/sdk/index';
import puppeteer from 'puppeteer';
import { createInterface } from 'node:readline';
import content from '../../rlvr/content.json';
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
let browser: Awaited<ReturnType<typeof puppeteer.launch>> | undefined;
let lastId = 0;
const harvests = new Map(content.harvests.flatMap(row => row.loc_ids.map(id => [id, row] as const)));
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
    max_ticks: number;
    name: string;
    arguments: Record<string, unknown>;
}

async function action(name: string, args: Record<string, unknown>, deadline: number, tickBudget: number): Promise<ActionOutcome> {
    const start = performance.now();
    const before = await engine('state');
    if (!before.paused) throw new Error('Decision boundary is not paused');
    let status: ActionOutcome['status'] = 'failed';
    let reason = 'precondition';
    let target: Resource | null = null;
    const maxTicks = Math.min(tickBudget, name === 'wait' ? Number(args.ticks) : name === 'attackNpc' ? 10 : 35);
    let rule: typeof content.harvests[number] | undefined;
    if (name === 'harvest') {
        target = before.resources.find(r => r.id === args.id && r.x === args.x && r.z === args.z) ?? null;
        rule = target ? harvests.get(target.id) : undefined;
    } else if (name === 'attackNpc') {
        target = before.npcs.find(n => n.index === args.index) ?? null;
    }
    if ((name === 'harvest' || name === 'attackNpc') && !target) {
        const after = await engine('state');
        return { name, status, reason: 'target_not_visible', before, after, target, elapsed_seconds: (performance.now() - start) / 1000 };
    }
    await engine(`resume?ticks=${maxTicks}`, 'POST');
    try {
        let accepted = false;
        if (name === 'harvest' && target && rule) {
            accepted = (await sdk.sendInteractLoc(target.x, target.z, target.id, 1)).success;
        } else if (name === 'attackNpc' && target) {
            const npc = sdk.getState()?.nearbyNpcs.find(n => n.index === target!.index);
            const option = npc?.optionsWithIndex.find(o => o.text.toLowerCase() === 'attack');
            if (option) accepted = (await sdk.sendInteractNpc(target.index!, option.opIndex)).success;
        } else if (name === 'walkTo') {
            accepted = (await sdk.sendWalk(Number(args.x), Number(args.z))).success;
        } else if (name === 'equipItem' || name === 'eatFood') {
            const item = sdk.getInventory().find(i => i.slot === args.slot);
            const option = item?.optionsWithIndex.find(o => name === 'equipItem' ? /^(wield|wear|equip)$/i.test(o.text) : /^eat$/i.test(o.text));
            if (item && option) accepted = (await sdk.sendUseItem(item.slot, option.opIndex)).success;
        } else if (name === 'setCombatStyle') {
            accepted = (await sdk.sendSetCombatStyle(Number(args.style))).success;
        } else if (name === 'wait') {
            accepted = true;
        } else throw new Error('Unsupported action');
        if (accepted) {
            try {
                const state = await until(s => {
                    if (s.tick <= before.tick) return false;
                    if (s.hp <= 0 || !s.in_game || s.tick - before.tick >= maxTicks) return true;
                    if (name === 'harvest' && rule) return count(s, rule.item_id) > count(before, rule.item_id);
                    if (name === 'walkTo') return s.position[0] === args.x && s.position[1] === args.z;
                    if (name === 'equipItem') return JSON.stringify(s.equipment) !== JSON.stringify(before.equipment);
                    if (name === 'eatFood') return s.inventory.length < before.inventory.length;
                    if (name === 'setCombatStyle') return s.combat_style === args.style;
                    return false;
                }, deadline - 2000);
                status = 'completed';
                reason = state.hp <= 0 ? 'death' : 'action_window_complete';
            } catch (error) {
                if (!(error instanceof ActionBudgetReached)) throw error;
                reason = 'action_budget';
            }
        } else reason = 'interaction_rejected';
        // Harvesting and walking stop at their boundary. Combat persists while paused.
        if (name === 'harvest' || name === 'walkTo') await engine('cancel', 'POST');
    } finally {
        await engine('pause', 'POST');
    }
    const after = await engine('state');
    if (rule) {
        status = count(after, rule.item_id) > count(before, rule.item_id) ? 'completed' : 'failed';
        reason = status === 'completed' ? 'harvested' : 'no_yield';
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
    send({ version: 2, episode_id: episode, world_id: world, request_id: '0', status: 'ready', state });
    for await (const line of createInterface({ input: process.stdin })) {
        let requestId = '';
        try {
            const cmd: Command = JSON.parse(line);
            requestId = cmd.request_id;
            if (
                cmd.version !== 2 ||
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
                if (!Number.isInteger(cmd.max_ticks) || cmd.max_ticks < 1 || cmd.max_ticks > 35) throw new Error('Invalid tick budget');
                task = action(cmd.name, cmd.arguments, deadline, cmd.max_ticks);
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
                version: 2,
                request_id: cmd.request_id,
                episode_id: episode,
                world_id: world,
                status: 'ok',
                result,
            });
        } catch (_error) {
            diagnose(_error);
            send({
                version: 2,
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
        version: 2,
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
