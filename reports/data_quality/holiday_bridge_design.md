# Holiday Bridge Design

## Why holidays are mapped by store

Holiday scope is geographic. National events apply to every store, regional
events apply only to stores in the matching state, and local events apply only
to stores in the matching city. Mapping holidays to `store_nbr` before loading
the warehouse preserves those rules and produces a clear daily store grain.

## Why joining only by date is incorrect

A date-only join would treat every regional or local event as if it applied to
every store. That would incorrectly assign holidays across cities and states,
distort holiday comparisons, and make local effects appear national.

## Why the bridge is separate from the sales fact

Sales uses the finer `date_key + store_key + family_key` grain, while holiday
mapping uses `date_key + store_key`. Keeping holiday text in a separate bridge
prevents a store-day event from being repeated once per product family. It also
keeps long descriptions, types, and locales out of the numeric sales fact.
Consumers can aggregate sales to the store-day grain before joining the bridge,
or use the dimension keys with an explicitly controlled semantic model.
