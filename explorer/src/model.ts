export interface Point {offset:number;session:string;ar:number;car:number;lower:number;upper:number}
export interface EventRow {
  id:string;symbol:string;title:string;input_kind:string;event_type:string;
  original_date:string;timestamp:string|null;timezone:string;timestamp_basis:string;
  source_url:string|null;source_note:string;announcement_status:string;
  aligned_session:string|null;alignment:string;status:string;reason:string|null;
  n_estimation:number;n_event:number;estimation_start?:string;estimation_end?:string;
  event_start?:string;event_end?:string;car:number|null;path:Point[];limitations:string;
}
export interface Evidence {schema_version:number;run_id:string;events:EventRow[];limitations:string}
export interface Filters {kind:string;symbol:string;type:string;status:string;start:string;end:string}
export function filterEvents(rows:EventRow[],f:Filters):EventRow[] {
  return rows.filter(e=>e.input_kind===f.kind && (!f.symbol||e.symbol===f.symbol)
    && (!f.type||e.event_type===f.type) && (!f.status||e.status===f.status)
    && (!f.start||(e.aligned_session??e.original_date)>=f.start)
    && (!f.end||(e.aligned_session??e.original_date)<=f.end));
}
export function parseEvidence(value:unknown):Evidence {
  if (!value || typeof value!=='object') throw new Error('Invalid evidence file');
  const v=value as Evidence;
  if (v.schema_version!==1 || !Array.isArray(v.events)|| typeof v.run_id!=='string')
    throw new Error('Unsupported evidence schema');
  for (const e of v.events) {
    if (typeof e.id!=='string'||typeof e.symbol!=='string'||!Array.isArray(e.path)
      || !['historical','synthetic','live'].includes(e.input_kind)
      || !['included','excluded'].includes(e.status)) throw new Error('Invalid event record');
    for (const p of e.path) if (![p.offset,p.ar,p.car,p.lower,p.upper].every(Number.isFinite))
      throw new Error('Invalid curve values');
  }
  return v;
}
