create table hp (
    model text not null,
    clip_thr float not null,
    n_client int not null,
    split text not null,
    ordinal int not null,
    round int not null,
    accuracy int not null,
    primary key (model, clip_thr, n_client, split, ordinal, round)
)