create view class as
WITH alltime as (
    SELECT
        algorithm,
        split,
        ordinal,
        time_agg_update + time_enc + time_dec AS time_crypto,
        time_local_train AS time_train,
        time_calc_mask, time_agg_mask AS time_partition
    FROM
        trains
    where
        strategy = 'max'
),
sumtime as (
    select
        algorithm,
        split,
        ordinal,
        sum(time_crypto) as time_crypto,
        sum(time_train) as time_train,
        sum(time_partition) as time_partition
    from alltime
    group by algorithm, split, ordinal
)
select
    algorithm,
    split,
    avg(time_crypto) as time_crypto,
    avg(time_train) as time_train,
    avg(time_partition) as time_partition
from
    sumtime
group by
    algorithm, split
ORDER BY
    algorithm ASC, split DESC

