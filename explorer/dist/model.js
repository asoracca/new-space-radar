export function filterEvents(rows, f) {
    return rows.filter(e => e.input_kind === f.kind && (!f.symbol || e.symbol === f.symbol)
        && (!f.type || e.event_type === f.type) && (!f.status || e.status === f.status)
        && (!f.start || (e.aligned_session ?? e.original_date) >= f.start)
        && (!f.end || (e.aligned_session ?? e.original_date) <= f.end));
}
export function parseEvidence(value) {
    if (!value || typeof value !== 'object')
        throw new Error('Invalid evidence file');
    const v = value;
    if (v.schema_version !== 1 || !Array.isArray(v.events) || typeof v.run_id !== 'string')
        throw new Error('Unsupported evidence schema');
    for (const e of v.events) {
        if (typeof e.id !== 'string' || typeof e.symbol !== 'string' || !Array.isArray(e.path)
            || !['historical', 'synthetic', 'live'].includes(e.input_kind)
            || !['included', 'excluded'].includes(e.status))
            throw new Error('Invalid event record');
        for (const p of e.path)
            if (![p.offset, p.ar, p.car, p.lower, p.upper].every(Number.isFinite))
                throw new Error('Invalid curve values');
    }
    return v;
}
