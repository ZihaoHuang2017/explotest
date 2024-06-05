COMBY_M="$(cat <<"MATCH"
for :[contained] in :[container]:
MATCH
)"

COMBY_R="$(cat <<"REWRITE"
for index_:[contained].offset.start_reserved, (:[contained]) in enumerate(:[container]):
REWRITE
)"

COMBY_RULE="$(cat <<"RULE"
where 
match :[container] {
| "enumerate(:[_])" -> false
| "range(:[_])" -> false
| ":[_]" -> true
} 

RULE
)"

comby "$COMBY_M" "$COMBY_R" -rule "$COMBY_RULE" .py -stats -match-newline-at-toplevel