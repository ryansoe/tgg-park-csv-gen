<!-- 50bec22b-2858-4f25-a9ed-19d62ffb00f6 d3510ff7-9d41-4c97-9820-9eecd55ffe1f -->
# Fix Cumulative Turn Hiding Bug

## Root Cause Identified

After investigating why turns still don't appear, I found **the critical bug** in `build_directions()` (lines 564-586):

When edges are merged, the code updates the segment's bearing:
```python
else:
    # Merge into current segment
    current_seg.length_m += edge.length_m
    current_seg.bearing = edge.bearing  # ← BUG!
```

**The Problem**: Each new edge is compared to the UPDATED bearing from the previous merge, not the ORIGINAL segment start bearing. This hides cumulative turns:

- Start: bearing 90° (East)
- Edge 1: bearing 100° → Δ10° vs 90° → no turn → merge, **bearing now 100°**
- Edge 2: bearing 110° → Δ10° vs **100°** → no turn → merge, **bearing now 110°**
- Edge 3: bearing 120° → Δ10° vs **110°** → no turn → merge
- **Result**: One long "continue" segment, but we actually turned 30°!

## The Fix

**File**: `walking_tour_generator.py`, lines 556-586

**Option 1 (Simplest)**: Don't update the bearing when merging - keep the original segment bearing for all comparisons:

```python
else:
    # Merge into current segment
    current_seg.length_m += edge.length_m
    # REMOVE: current_seg.bearing = edge.bearing
    # Keep original bearing for accurate cumulative turn detection
```

**Option 2 (Better)**: Track both start and end bearings separately, compare using start bearing:

Add `start_bearing` field to EdgeInfo and compare against it instead of the continuously-updated `bearing` field.

I recommend **Option 1** for simplicity and immediate impact.

## Implementation Steps

1. Remove the bearing update line in the merge block (line 584)
2. Test with debug script to verify turns are detected
3. Re-generate Boston College tours 1-5
4. Verify turns now appear in output

## Previous Improvements (Already Completed)

- ✓ Tightened turn detection: 25°→15°, 35°→30°, 100°→90°
- ✓ Added initial cardinal direction to first step
- ✓ Strengthened LLM prompts to preserve turns
- ✓ Added turn count validation
- ✓ Reduced segment merging aggressiveness

These improvements will NOW work correctly once the cumulative turn bug is fixed.

### To-dos

- [ ] Remove bearing update in merge block (line 584) to fix cumulative turn hiding
- [ ] Run debug script to verify turns are now detected in raw output
- [ ] Re-generate Boston College tours 1-5 with fix applied
- [ ] Verify all expected turns now appear in generated tours