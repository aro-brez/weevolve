#!/bin/bash
# WeEvolve Post-Task Learning Evaluation Hook
# Pattern from Claudeception — triggers after every user prompt
# Reminds the agent to evaluate whether extractable knowledge was produced

echo "SEED PHASE 8 (IMPROVE): After completing this task, evaluate:"
echo "  1. Did this require non-obvious investigation or debugging?"
echo "  2. Was the solution something that would help in future similar situations?"
echo "  3. Is there a reusable pattern worth saving as a skill?"
echo "  4. Should the collective (8 Owls) know about this discovery?"
echo ""
echo "If YES to any: run 'weevolve learn --text \"[discovery]\"' to persist the knowledge."
