import fs from 'fs';
import RibbonServer from './reward_server';

const DIR = 'F:/Games/World_of_Warships_ASIA';
const MOD_DIR = [DIR, 'bin/2697511/res_mods/0.9.6.1/PnFMods/RLMod'].join('/');
const REPLAY_DIR = [DIR, 'replays'].join('/');

const PREFIX = {
    Battle: 'battle',
    Ribbon: 'ribbon'
};

const POSTFIX = {
    Replay: '.wowsreplay'
};

const filenames = [];

const server = new RibbonServer();

fs.watch(MOD_DIR, (eventType, filename) => {
    if (!filename) { return; }
    if (filename.startsWith(PREFIX.Battle)) {
        server.resetRibbons();
        return;
    }
    if (!filename.startsWith(PREFIX.Ribbon)) { return; }
    if (eventType != 'change') { return; }

    if (filenames.includes(filename)) { return; }
    filenames.push(filename);

    const path = [MOD_DIR, filename].join('/');

    const data = JSON.parse(fs.readFileSync(path).toString('utf-8'));
    console.log(Date.now().toString() + ': ' + data.ribbons);
    data.ribbons.forEach(ribbon => {
        server.addRibbon(ribbon);
    });

    fs.unlink(path, () => {
        const fileIndex = filenames.indexOf(filename);
        if (fileIndex > -1) {
            filenames.splice(fileIndex, 1);
        }
    });
});

fs.watch(REPLAY_DIR, (eventType, filename) => {
    if (!filename.endsWith(POSTFIX.Replay)) {
        return;
    }

    if (eventType == 'change') {
        const path = [REPLAY_DIR, filename].join('/');
        const replay_dir = [__dirname, '..', 'replays'].join('/');
        const dir_head = filename.split('_').shift();
        const destination = [replay_dir, dir_head, filename].join('/');
        if (!fs.existsSync(replay_dir)) {
            fs.mkdirSync(replay_dir);
        }
        const target_dir = [replay_dir, dir_head].join('/');
        if (!fs.existsSync(target_dir)) {
            fs.mkdirSync(target_dir);
        }
        fs.copyFile(path, destination, err => {
            if (err) {
                //console.error(err);
            }
            console.log('Succeed to copy replay file:', destination);
        });
    }
});

server.start();

//

import * as grpc from 'grpc';
import * as protoLoader from '@grpc/proto-loader';

Array.prototype.empty = () => this.length == 0;

const PROTO_PATH = __dirname + '/../protos/ribbon.proto';

const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true
});

const protoDescriptor = grpc.loadPackageDefinition(packageDefinition);

const RibbonService = protoDescriptor.wows.RibbonService;

class RibbonServer {

    constructor() {
        this.ribbons = [];
        this.server = new grpc.Server();
        this.server.addService(RibbonService.service, {
            getRibbon: this.getRibbon,
            __ribbons: this.ribbons
        });
        this.server.bind('0.0.0.0:61084', grpc.ServerCredentials.createInsecure());
    }

    start() {
        console.log('RibbonServer is running..');
        this.server.start();
    }

    addRibbon(ribbon) {
        this.ribbons.push(ribbon);
        console.log('ribbons:', this.ribbons);
    }

    resetRibbons() {
        this.ribbons.splice(0, this.ribbons.length);
    }

    getRibbon(call, callback) {
        const id = this.__ribbons.empty() ? 0 : this.__ribbons.shift();
        callback(null, { id });
    }
}

export default RibbonServer;