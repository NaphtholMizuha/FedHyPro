CREATE VIEW result AS
WITH trains_view AS (
    SELECT
        *,
        (time_agg_update + time_enc + time_dec) AS time_crypt,
        (time_local_train + time_calc_mask + time_agg_mask) AS time_common
    FROM
        trains
),
total_time_view AS (
    SELECT
        *,
        (time_crypt + time_common) AS total_time
    FROM
        trains_view
),
selected_columns_view AS (
    SELECT
        algorithm,
        strategy,
        split,
        ordinal,
        round,
        accuracy,
        total_time
    FROM
        total_time_view
    ORDER BY
        algorithm, strategy, split, ordinal, round
),
grouped_view AS (
    SELECT
        algorithm,
        strategy,
        split,
        ordinal,
        MAX(accuracy) AS max_accuracy,
        SUM(total_time) AS total_time_sum
    FROM
        selected_columns_view
    GROUP BY
        algorithm, strategy, split, ordinal
),
final_grouped_view AS (
    SELECT
        algorithm,
        strategy,
        split,
        AVG(max_accuracy) AS acc,
        AVG(total_time_sum) AS time
    FROM
        grouped_view
    GROUP BY
        algorithm, strategy, split
),
formatted_result_view AS (
    SELECT
        algorithm,
        strategy,
        split,
        ROUND(acc * 100, 2) AS acc,
        ROUND(time) AS time
    FROM
        final_grouped_view
    ORDER BY
        algorithm ASC,
        split DESC
)
SELECT * FROM formatted_result_view;

create view clip_thr AS
with temp as (
    select clip_thr, ordinal, max(accuracy) as acc
    from hp
    where split='iid'
    GROUP BY model, split, clip_thr, ordinal
)
select clip_thr, round(avg(acc), 2) as accuracy
from temp
group by clip_thr;

drop view if exists clip_thr;
create view clip_thr AS
with temp as (
    select clip_thr, ordinal, max(accuracy) as acc
    from hp
    where split='iid' and n_client=10
    GROUP BY model, split, clip_thr, ordinal
)
select clip_thr, round(avg(acc) * 100, 2) as accuracy
from temp
group by clip_thr;

drop view if exists n_client;

create view n_client AS
with temp as (
    select n_client, ordinal, max(accuracy) as acc
    from hp
    where split='iid' and clip_thr=10
    GROUP BY model, split, n_client, ordinal
)
select n_client, round(avg(acc) * 100, 2) as accuracy
from temp
group by n_client;