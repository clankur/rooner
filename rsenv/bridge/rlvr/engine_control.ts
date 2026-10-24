// Installed into a private, pinned engine copy. Never loaded by normal servers.
import fs from 'fs';
import World from '#/engine/World.js';
import Player from '#/engine/entity/Player.js';
import InvType from '#/cache/config/InvType.js';
import ObjType from '#/cache/config/ObjType.js';
import LocType from '#/cache/config/LocType.js';
import VarPlayerType from '#/cache/config/VarPlayerType.js';

export interface Item {
    slot: number;
    id: number;
    count: number;
}

export interface Resource {
    id: number;
    x: number;
    z: number;
    name: string;
}

interface ResetFixture {
    position: [number, number, number];
    xp_tenths: number[];
    levels: number[];
    inventory: Item[];
    tutorial: number;
}

export interface Observation {
    episode_id: string;
    world_id: string;
    sequence: number;
    tick: number;
    position: [number, number, number];
    xp_tenths: number[];
    levels: number[];
    inventory: Item[];
    equipment: Item[];
    bank: Item[];
    hp: number;
    pending: boolean;
    tutorial: number;
    resources: Resource[];
    in_game: boolean;
    runtime: {
        engine_rss_bytes: number;
        engine_cpu_seconds: number;
        wall_seconds: number;
    };
}

const fixture: ResetFixture = JSON.parse(fs.readFileSync(process.env.RLVR_FIXTURE!, 'utf8'));
const username = process.env.RLVR_USERNAME!;
const token = process.env.RLVR_CONTROL_TOKEN!;
let sequence = 0;
let bootstrapped = false;

export function initializePlayer(player: Player): void {
    if (player.username !== username) throw new Error('RLVR worlds admit only their leased account');
    if (ObjType.getId('logs') !== 1511 || ObjType.getId('bronze_axe') !== 1351) {
        throw new Error('Pinned resource IDs do not match the engine');
    }
    player.x = fixture.position[0];
    player.z = fixture.position[1];
    player.level = fixture.position[2];
    for (let i = 0; i < 21; i++) {
        player.stats[i] = fixture.xp_tenths[i];
        player.levels[i] = fixture.levels[i];
        player.baseLevels[i] = fixture.levels[i];
    }
    player.setVar(VarPlayerType.getId('tutorial'), fixture.tutorial);
    for (const name of ['inv', 'worn', 'bank']) {
        player.invClear(InvType.getId(name));
    }
    const inv = player.getInventory(InvType.INV)!;
    for (const item of fixture.inventory) inv.set(item.slot, { id: item.id, count: item.count });
    player.runenergy = 10000;
}

function player(): Player {
    const p = World.getPlayerByUsername(username);
    if (!p) throw new Error('Leased player not logged in');
    return p;
}

function cancel(p: Player): void {
    p.stopAction();
    p.clearWaypoints();
    p.playAnimation(-1, 0);
}

function snapshot(): Observation {
    const p = player();
    const items = (name: string): Item[] =>
        p
            .getInventory(InvType.getId(name))!
            .items.flatMap((item, slot) => (item ? [{ slot, id: item.id, count: item.count }] : []));
    const resources: Resource[] = [];
    const zones = new Set<number>();
    for (let x = p.x - 16; x <= p.x + 16; x++) {
        for (let z = p.z - 16; z <= p.z + 16; z++) {
            const key = ((x >> 3) << 16) | (z >> 3);
            if (zones.has(key)) continue;
            zones.add(key);
            for (const loc of World.gameMap.getZone(x, z, p.level).getAllLocsSafe()) {
                if (Math.max(Math.abs(loc.x - p.x), Math.abs(loc.z - p.z)) > 16) continue;
                const type = LocType.get(loc.type);
                if (type.name === 'Tree') resources.push({ id: loc.type, x: loc.x, z: loc.z, name: type.name });
            }
        }
    }
    resources.sort((a, b) => a.x - b.x || a.z - b.z || a.id - b.id);
    return {
        episode_id: process.env.RLVR_EPISODE_ID!,
        world_id: process.env.RLVR_WORLD_ID!,
        sequence: ++sequence,
        tick: World.currentTick,
        position: [p.x, p.z, p.level],
        xp_tenths: Array.from(p.stats),
        levels: Array.from(p.levels),
        inventory: items('inv'),
        equipment: items('worn'),
        bank: items('bank'),
        hp: p.levels[3],
        pending: p.hasInteraction() || p.waypointIndex >= 0 || p.activeScript !== null,
        tutorial: p.getVar(VarPlayerType.getId('tutorial')) as number,
        resources,
        in_game: p.slot >= 0,
        runtime: {
            engine_rss_bytes: process.memoryUsage().rss,
            engine_cpu_seconds: (process.cpuUsage().user + process.cpuUsage().system) / 1e6,
            wall_seconds: performance.now() / 1000,
        },
    };
}

export function startControl(): void {
    if (!token || !process.env.RLVR_EPISODE_ID) throw new Error('Missing private RLVR identity');
    Bun.serve({
        hostname: '127.0.0.1',
        port: Number(process.env.RLVR_CONTROL_PORT),
        fetch(req: Request): Response {
            if (req.headers.get('authorization') !== `Bearer ${token}`) return new Response('', { status: 403 });
            try {
                const path = new URL(req.url).pathname;
                if (req.method === 'GET' && path === '/health') return Response.json({ ready: true });
                if (req.method === 'POST' && path === '/bootstrap') {
                    if (bootstrapped) throw new Error('Bootstrap is single-use; reset the world');
                    const p = player();
                    cancel(p);
                    p.activeScript = null;
                    p.queue.clear();
                    p.weakQueue.clear();
                    p.engineQueue.clear();
                    // Login timers remain real engine timers, reset to a common episode origin.
                    for (const timer of p.timers.values()) timer.clock = World.currentTick;
                    initializePlayer(p);
                    bootstrapped = true;
                    return Response.json(snapshot());
                }
                if (!bootstrapped) throw new Error('Episode not bootstrapped');
                if (req.method === 'POST' && path === '/cancel') {
                    cancel(player());
                    return Response.json(snapshot());
                }
                if (req.method === 'GET' && path === '/state') return Response.json(snapshot());
                return new Response('', { status: 404 });
            } catch (err) {
                return Response.json({ error: String(err) }, { status: 409 });
            }
        },
    });
}
