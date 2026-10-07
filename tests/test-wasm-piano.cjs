// Exercise the real Embind exports; no model or audio weights required.
const assert = require('node:assert/strict');
const path = require('node:path');
const factory = require(path.resolve(process.argv[2]));
factory().then(module => {
    assert.equal(module.sessionPianoSampleRate(), 0);
    assert.deepEqual(module.sessionPianoNotes(new Float32Array()), []);
    assert.deepEqual(module.sessionPianoNotes(new Float32Array(160)), []);
    console.log('WASM piano bindings: 3 assertions passed');
}).catch(error => {
    console.error(error);
    process.exitCode = 1;
});
