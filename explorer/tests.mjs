import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {filterEvents,parseEvidence} from './dist/model.js';
const evidence=parseEvidence(JSON.parse(readFileSync(new URL('../evidence/results.json',import.meta.url))));
const base={kind:'historical',symbol:'',type:'',status:'',start:'',end:''};
test('saved historical default and explicit synthetic mode',()=>{
 assert.equal(filterEvents(evidence.events,base).length,18);
 const synthetic=filterEvents(evidence.events,{...base,kind:'synthetic'});
 assert.equal(synthetic.length,2);assert.ok(synthetic.every(e=>e.path.length===8));
});
test('combined filters use aligned date',()=>{
 const rows=filterEvents(evidence.events,{...base,symbol:'LUNR',type:'launch',start:'2024-02-23',end:'2024-02-23'});
 assert.equal(rows.length,1);assert.equal(rows[0].id,'lunr-001');
 assert.equal(filterEvents(evidence.events,{...base,status:'included'}).length,0);
});
test('invalid schema and curve rejected',()=>{
 assert.throws(()=>parseEvidence({schema_version:2,events:[]}));
 const bad=structuredClone(evidence);bad.events.find(e=>e.path.length).path[0].car=Infinity;
 assert.throws(()=>parseEvidence(bad));
});
