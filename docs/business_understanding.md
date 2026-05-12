# Task 1 — Business Understanding
# EV Battery Health Prediction
# Client: BMW Group — Electric Vehicle Division

---

## The Client & Business Problem

**Client:** BMW Group — Electric Vehicle Division
**Sector:** Automotive / Electric Vehicles
**Location:** Munich, Germany

### The Problem

BMW Group sells over 200,000 electric vehicles per year
across the iX, i4, i5 and i7 model lines. Each EV battery
pack costs between 8,000 and 15,000 euros to replace.

Three critical business challenges:

**1. Range Anxiety**
Drivers do not trust their battery range estimate.
Current systems use simple capacity lookup tables
that become inaccurate as batteries age.

**2. Unexpected Battery Failures**
Some batteries degrade faster than expected.
Failures under warranty cost BMW heavily.
Each unexpected failure costs 8,000-15,000 euros.

**3. Premature Replacements**
Fixed replacement schedules replace healthy batteries.
This wastes money and creates unnecessary e-waste.

### Our Solution — Data-Driven Battery Management

Use real discharge sensor data to:
1. Predict State of Health (SoH) in real-time
2. Predict Remaining Useful Life (RUL) in cycles
3. Classify battery condition: Healthy / Degraded / Near EOL

---

## Key Definitions

**State of Health (SoH)**
= Current Capacity / Rated Capacity
= How much of its original capacity the battery still has
= 100% when new, 70% at End of Life (industry standard)

**Remaining Useful Life (RUL)**
= Cycles remaining before SoH drops to 70%
= Tells BMW exactly when to schedule replacement

**End of Life (EOL)**
= When capacity drops 30% below rated value
= 2.0 Ah rated → 1.4 Ah at EOL
= Industry standard threshold

---

## Dataset: NASA Li-ion Battery Aging

Source: NASA Ames Prognostics Center of Excellence
Same dataset used in real research at BMW, Tesla, Siemens

4 batteries charged and discharged until failure:
- B0005, B0006, B0007, B0018
- Each battery: hundreds of charge/discharge cycles
- Measurements: Voltage, Current, Temperature per cycle

---

## Expected Business Impact

Accurate SoH prediction enables:
- Real-time accurate range display for drivers
- Proactive battery replacement before failures
- 25-40% reduction in unnecessary replacements
- Significant reduction in warranty claim costs
- Better driver trust in BMW electric vehicles
