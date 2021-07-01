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