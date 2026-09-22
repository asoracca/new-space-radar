import {filterEvents,parseEvidence,type EventRow,type Evidence} from './model.js';
const byId=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const select=(id:string)=>byId<HTMLSelectElement>(id);
const text=(id:string,value:string)=>{byId(id).textContent=value;};
let evidence:Evidence;
let chosen='lunr-001';
function options(id:string,values:string[]):void {
  const el=select(id), old=el.value;
  el.replaceChildren(new Option('All',''),...values.map(x=>new Option(x,x)));
  if(values.includes(old)) el.value=old;
}
function populate():void {
  const rows=evidence.events.filter(e=>e.input_kind===select('kind').value);
  options('symbol',[...new Set(rows.map(e=>e.symbol))].sort());
  options('type',[...new Set(rows.map(e=>e.event_type))].sort());
}
const percent=(n:number|null)=>n===null?'Not estimated':`${(n*100).toFixed(3)}%`;
function render():void {
  const rows=filterEvents(evidence.events,{kind:select('kind').value,symbol:select('symbol').value,
    type:select('type').value,status:select('status').value,
    start:byId<HTMLInputElement>('start').value,end:byId<HTMLInputElement>('end').value});
  text('counts',`${rows.length} matching events · ${rows.filter(e=>e.status==='included').length} included · ${rows.filter(e=>e.status==='excluded').length} excluded`);
  const body=byId('rows');body.replaceChildren();
  for(const e of rows){
    const tr=document.createElement('tr');
    const td=document.createElement('td');const button=document.createElement('button');
    button.textContent=`${e.symbol} · ${e.title}`;
    button.onclick=()=>{chosen=e.id;render();};
    button.setAttribute('aria-pressed',String(e.id===chosen));td.append(button);tr.append(td);
    for(const value of [e.aligned_session??e.original_date,e.status,percent(e.car)]){
      const cell=document.createElement('td');cell.textContent=value;tr.append(cell);
    }
    body.append(tr);
  }
  const e=rows.find(e=>e.id===chosen)??rows[0];
  byId('detail').hidden=!e;
  text('empty',e?'':'No events match these filters.');
  if(e) {chosen=e.id;detail(e);}
}
function detail(e:EventRow):void {
  text('title',e.title);text('badge',`${e.input_kind.toUpperCase()} · ${e.symbol} · ${e.event_type}`);
  text('alignment',`${e.timestamp??'Time unknown'} (${e.timezone}) → ${e.aligned_session??'Not aligned'}`);
  text('timing',`Time basis: ${e.timestamp_basis}. Announcement status: ${e.announcement_status}.`);
  text('sample',`Estimation: ${e.n_estimation} observations${e.estimation_start?` · ${e.estimation_start} to ${e.estimation_end}`:''}. Event: ${e.n_event} observations${e.event_start?` · ${e.event_start} to ${e.event_end}`:''}.`);
  text('result',e.reason?`Excluded: ${e.reason.replaceAll('_',' ')}`:`CAR [-2,+5]: ${percent(e.car)} · null baseline: 0.000%`);
  text('source-note',e.source_note);text('limitations',e.limitations);
  const link=byId<HTMLAnchorElement>('source');
  link.hidden=!e.source_url;
  if(e.source_url && /^https?:\/\//.test(e.source_url)){link.href=e.source_url;link.textContent=e.input_kind==='synthetic'?'Synthetic generator source':'Original source';}
  else {link.removeAttribute('href');link.hidden=true;}
  byId('curve').replaceChildren();
  if(!e.path.length){text('curve-note','No curve: this event does not have eligible return observations.');return;}
  text('curve-note','CAR is accumulated from day −2. Dashed line: zero baseline. Table shows pointwise model intervals; these are not trading probabilities.');
  const ns='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 680 260');
  svg.setAttribute('role','img');svg.setAttribute('aria-label',`Aligned CAR curve for ${e.symbol}`);
  const min=Math.min(0,...e.path.map(p=>p.car))-.005,max=Math.max(0,...e.path.map(p=>p.car))+.005;
  const x=(i:number)=>65+i*580/(e.path.length-1||1),y=(v:number)=>210-(v-min)/(max-min)*180;
  function line(points:string,color:string,dash=false){const p=document.createElementNS(ns,'polyline');p.setAttribute('points',points);p.setAttribute('fill','none');p.setAttribute('stroke',color);p.setAttribute('stroke-width','3');if(dash)p.setAttribute('stroke-dasharray','5 5');svg.append(p);}
  function label(px:number,py:number,t:string){const el=document.createElementNS(ns,'text');el.setAttribute('x',String(px));el.setAttribute('y',String(py));el.setAttribute('fill','#425367');el.setAttribute('font-size','14');el.textContent=t;svg.append(el);}
  line(`${x(0)},${y(0)} ${x(e.path.length-1)},${y(0)}`,'#8b99a8',true);
  line(e.path.map((p,i)=>`${x(i)},${y(p.car)}`).join(' '),'#006d77');
  e.path.forEach((p,i)=>label(x(i)-5,240,String(p.offset)));
  label(0,25,percent(max));label(0,210,percent(min));label(275,258,'Trading-session offset');
  byId('curve').append(svg);
  const table=document.createElement('table');
  const caption=document.createElement('caption');caption.textContent='Exact curve values and 95% pointwise IID model intervals';table.append(caption);
  const head=document.createElement('tr');
  for(const name of ['Offset','Session','AR','CAR','Lower','Upper']){const cell=document.createElement('th');cell.textContent=name;cell.scope='col';head.append(cell);}table.append(head);
  for(const p of e.path){const tr=document.createElement('tr');for(const value of [String(p.offset),p.session,percent(p.ar),percent(p.car),percent(p.lower),percent(p.upper)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);}
  byId('curve').append(table);
}
async function start(){
  try{
    const response=await fetch('../evidence/results.json');if(!response.ok)throw new Error(`Evidence request failed (${response.status})`);
    evidence=parseEvidence(await response.json());text('run',`Saved run ${evidence.run_id.slice(0,12)}`);
    populate();render();
    for(const id of ['kind','symbol','type','status','start','end'])byId(id).addEventListener('change',()=>{if(id==='kind')populate();render();});
  }catch(error){text('counts',`Unable to load saved evidence: ${String(error)}. Serve the repository root with Python's local HTTP server.`);}
}
void start();
