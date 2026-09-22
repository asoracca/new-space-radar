-- Bind :run_id, :symbol, :event_type, :start, :end in your SQLite client.
SELECT e.symbol,e.event_type,e.source_url,r.aligned_session,r.status,r.reason,r.car
FROM events e JOIN results r ON r.event_id=e.id
WHERE r.run_id=:run_id AND e.symbol=:symbol AND e.event_type=:event_type
AND COALESCE(r.aligned_session,e.original_date) BETWEEN :start AND :end;

SELECT e.input_kind,r.status,COUNT(*) AS n,AVG(r.car) AS mean_car
FROM events e JOIN results r ON r.event_id=e.id
WHERE r.run_id=:run_id GROUP BY e.input_kind,r.status;

SELECT reason,COUNT(*) FROM results WHERE run_id=:run_id AND status='excluded' GROUP BY reason;

-- CAR reconciliation: all differences should be zero to floating-point tolerance.
SELECT r.event_id,r.car-SUM(p.ar) AS difference FROM results r
JOIN points p USING(run_id,event_id) WHERE r.run_id=:run_id
GROUP BY r.event_id;
