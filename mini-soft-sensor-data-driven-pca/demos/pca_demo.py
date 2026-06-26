#!/usr/bin/env python3
"""
PCA Soft Sensor Demo — Python visualization companion.

This script demonstrates PCA concepts visually:
1. PCA dimensionality reduction on 2D data
2. Variance explained (scree plot)
3. T2 and SPE monitoring statistics

Run: python3 demos/pca_demo.py
Dependencies: numpy, matplotlib (optional)
"""
import sys
import math
import random

def demo_pca_2d():
    """Demonstrate PCA on 2D correlated data."""
    print("=" * 60)
    print("PCA Demo: Dimensionality Reduction on Correlated Data")
    print("=" * 60)

    # Generate correlated 2D data
    n = 50
    data = []
    for i in range(n):
        x = random.gauss(10.0, 2.0)
        y = 0.8 * x + random.gauss(0, 1.0)  # correlated
        data.append((x, y))

    # Center the data
    mean_x = sum(p[0] for p in data) / n
    mean_y = sum(p[1] for p in data) / n
    centered = [(x - mean_x, y - mean_y) for x, y in data]

    # Compute covariance matrix
    cov_xx = sum(x*x for x,y in centered) / (n-1)
    cov_xy = sum(x*y for x,y in centered) / (n-1)
    cov_yy = sum(y*y for x,y in centered) / (n-1)

    print(f"\nCovariance matrix: [[{cov_xx:.3f}, {cov_xy:.3f}], [{cov_xy:.3f}, {cov_yy:.3f}]]")

    # Eigen decomposition of 2x2 matrix
    # Characteristic equation: lambda^2 - trace*lambda + det = 0
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    disc = math.sqrt(trace*trace - 4*det)

    lambda1 = (trace + disc) / 2
    lambda2 = (trace - disc) / 2

    # Eigenvector for lambda1: (cov_xy, lambda1 - cov_xx)
    v1_x = cov_xy
    v1_y = lambda1 - cov_xx
    norm1 = math.sqrt(v1_x*v1_x + v1_y*v1_y)

    print(f"\nEigenvalues: lambda1={lambda1:.3f}, lambda2={lambda2:.3f}")
    print(f"PC1 direction: ({v1_x/norm1:.3f}, {v1_y/norm1:.3f})")
    print(f"Variance explained: PC1={lambda1/(lambda1+lambda2)*100:.1f}%, PC2={lambda2/(lambda1+lambda2)*100:.1f}%")

    # Project data onto PC1
    pc1_x = v1_x / norm1
    pc1_y = v1_y / norm1
    scores = [x*pc1_x + y*pc1_y for x,y in centered]

    print(f"\nScores (first 5): {scores[:5]}")
    print(f"Score variance: {sum(s*s for s in scores)/(n-1):.3f} (should match lambda1)")

    # Demonstrate soft sensor concept
    print("\n" + "=" * 60)
    print("Soft Sensor Concept: Predict Y from X via PC Regression")
    print("=" * 60)
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    x_test = 12.0

    # Simple linear regression: y = a*x + b
    mean_x_raw = sum(xs) / n
    mean_y_raw = sum(ys) / n
    num = sum((xs[i]-mean_x_raw)*(ys[i]-mean_y_raw) for i in range(n))
    den = sum((xs[i]-mean_x_raw)**2 for i in range(n))
    a = num / den
    b = mean_y_raw - a * mean_x_raw
    y_pred = a * x_test + b

    print(f"\nLinear regression: y = {a:.3f}*x + {b:.3f}")
    print(f"For x={x_test}, predicted y={y_pred:.3f}")
    print(f"(True relationship: y = 0.8*x + noise)")

    # PCR concept: using only PC1
    pcs_a = 1  # use 1 PC
    print(f"\nPCR with {pcs_a} PC(s):")
    print("  Step 1: Fit PCA on X (here just 1 variable)")
    print("  Step 2: Regress Y on PC scores")
    print("  Step 3: Project new X through PCA -> score -> Y_hat")
    print(f"  Result: prediction via latent variable model")

    # T2 and SPE demo
    print("\n" + "=" * 60)
    print("Process Monitoring Demo")
    print("=" * 60)

    # For n=50, A=1: T2_lim = 1*(49)*(51)/(50*49)*F(0.05,1,49) ~ 4.0
    print(f"\nTraining samples: {n}")
    print(f"Retained PCs: {pcs_a}")
    t2_lim = lambda1 * 4.0  # rough F-based threshold
    print(f"Approximate T2 limit (alpha=0.05): {t2_lim:.2f}")

    # Test a "normal" observation
    x_normal = 10.5
    y_normal = 0.8 * x_normal + 0.5  # within normal range
    x_c = x_normal - mean_x_raw
    score = x_c * pc1_x + (y_normal - mean_y_raw) * pc1_y
    t2_normal = score * score / lambda1
    print(f"\nNormal observation: x={x_normal}, y={y_normal:.1f}")
    print(f"  Score: {score:.3f}")
    print(f"  T2: {t2_normal:.3f} (limit: {t2_lim:.2f}) -> {'ALARM' if t2_normal > t2_lim else 'OK'}")

    # Test a "faulty" observation (breaks correlation)
    x_fault = 10.5
    y_fault = 5.0  # way off the correlation line
    x_c2 = x_fault - mean_x_raw
    score2 = x_c2 * pc1_x + (y_fault - mean_y_raw) * pc1_y
    # SPE = reconstruction error
    x_hat = score2 * pc1_x + mean_x_raw
    y_hat = score2 * pc1_y + mean_y_raw
    spe_fault = (x_fault - x_hat)**2 + (y_fault - y_hat)**2
    print(f"\nFaulty observation: x={x_fault}, y={y_fault}")
    print(f"  SPE: {spe_fault:.3f}")
    print(f"  (Large SPE indicates broken variable correlation)")
    print(f"  Diagnosis: y measurement likely faulty (sensor bias)")

    print("\n" + "=" * 60)
    print("Demo complete! See src/*.c for full C implementations.")
    print("=" * 60)

if __name__ == '__main__':
    random.seed(42)
    demo_pca_2d()
