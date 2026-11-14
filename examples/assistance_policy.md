Use the evaluation data field "annotation_status" as the routing rule. When it
is "needs_human", request human assistance and ask exactly one focused question
about the benchmark rule described by "missing_context". When it is "complete",
do not request human assistance. Always produce the best provisional pairwise
judgment before asking.

