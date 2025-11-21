# MixVPR Integration Status Report

## Summary
MixVPR code is **implemented but NOT fully integrated**. The system still relies on DBoW3 scores for filtering.

## Code Analysis

### ✅ What's Working:

1. **MixVPR Initialization** (lines 84-99 in neural_point.py):
   ```python
   self.use_mixvpr = cfg['tracking'].get('use_mixvpr', False)
   if self.use_mixvpr:
       self.loop_detector = LoopClosureDetector(...)
   ```

2. **MixVPR Query Method** (lines 176-203):
   - Detects loops using MixVPR
   - Returns results in DBoW-compatible format
   - Prints "MixVPR loop detected" when match found

3. **Frame Addition** (lines 155-169):
   - Adds frames to both DBoW3 AND MixVPR databases

### ⚠️ What's NOT Working:

**The Problem**: Line 684-688 ALWAYS uses `self.dbow_scores` for filtering:

```python
for s, dbow_score in zip(range(n_segments), self.dbow_scores):
    # ...
    if self.orb_filter:
        print(f"used dbow score for keyframe-{s} is {dbow_score}")  # <-- This prints!
        dbow_results = [x for x in dbow_results if x.Score > self.mult_dbow*dbow_score]
```

**Why it happens**:
- `compute_dbow_score()` is called unconditionally (line 1186)
- `self.dbow_scores` list is always populated with DBoW scores
- Loop closure filtering REQUIRES dbow_score threshold
- Even though `query_bow()` might return MixVPR results, the filtering uses DBoW thresholds

## Evidence from Logs

### Replica Run:
```
'use_mixvpr': True  # Config shows MixVPR enabled
used dbow score for keyframe-0 is 0.07593...  # But DBoW is being used!
```

### Why No "MixVPR loop detected" Messages?
Looking at logs - there are NO prints saying "MixVPR loop detected", which means:
1. Either `query_bow()` is not being called with frame_image parameter
2. Or MixVPR is not detecting any loops (scores below threshold)
3. Or the code path is not reaching the MixVPR branch

## Root Cause: Incomplete Integration

The MixVPR code exists but the FILTERING LOGIC still depends on DBoW scores:

```python
# Line 684: Always uses dbow_scores for filtering
for s, dbow_score in zip(range(n_segments), self.dbow_scores):
    # ...
    dbow_results = self.query_bow(features, frame_image=frame_image)  # May use MixVPR
    
    # But then filters using DBoW threshold!
    if x.Score > self.mult_dbow*dbow_score:  # <-- DBoW threshold applied to MixVPR!
```

This creates a hybrid system:
- MixVPR is queried
- But results are filtered using DBoW thresholds
- Which don't make sense for MixVPR scores (different scale/distribution)

## What Needs to be Fixed

### Option 1: Pure MixVPR (Recommended)
When `use_mixvpr: True`, skip DBoW filtering entirely:

```python
if self.use_mixvpr:
    # Use MixVPR results directly (no DBoW filtering)
    mixvpr_results = self.query_bow(features, frame_image=frame_image)
    # Process mixvpr_results without dbow_score threshold
else:
    # Traditional DBoW path
    for s, dbow_score in zip(range(n_segments), self.dbow_scores):
        # ... existing DBoW logic
```

### Option 2: Adaptive Thresholding
Compute "MixVPR scores" equivalent to dbow_scores:
- Track minimum MixVPR similarity scores between recent frames
- Use those as thresholds instead of dbow_scores

### Option 3: Hybrid System
Keep both but use appropriate thresholds for each:
```python
if self.use_mixvpr:
    threshold = self.mixvpr_threshold  # e.g., 0.85 cosine similarity
else:
    threshold = self.mult_dbow * dbow_score  # e.g., 0.07 DBoW score
```

## Verification Steps

### Check if MixVPR is being called:
```bash
grep "MixVPR loop detected" output/*/your_run.log
# Should show loop detections if MixVPR is working
```

### Check pretrained model:
```bash
ls -lh pretrained/mixvpr_resnet50.pth
# Should exist (~100MB)
```

### Add debug logging:
In neural_point.py line 690:
```python
print(f"DEBUG: use_mixvpr={self.use_mixvpr}, frame_image={'provided' if frame_image else 'None'}")
```

## Current Status per Dataset

| Dataset | Config | Logs Show | Actual Loop Detector |
|---------|--------|-----------|---------------------|
| Replica | use_mixvpr: True | "used dbow score" | DBoW3 |
| TUM fr1_desk | use_mixvpr: True | "used dbow score" (likely) | DBoW3 |
| TUM fr2_xyz | use_mixvpr: True | Running... | Unknown |
| TUM fr3_sit | use_mixvpr: True | Running... | Unknown |
| TUM fr3_walk | use_mixvpr: True | Running... | Unknown |

## Recommendation

**Stop current TUM runs and fix the integration before continuing**, OR let them complete to establish DBoW baseline, then fix and re-run.

The code structure is good, but needs one of the 3 fixes above to actually USE MixVPR instead of just calling it alongside DBoW.
