#!/usr/bin/env python3
import pandas as pd
import itertools
import argparse
from math import comb
from tqdm import tqdm
import csv

def find_violations(csv_path, output_path="violations.csv", max_violations=100, tol=1e-8):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Build symmetric distance map and adjacency for bipartite graph
    dist = {}
    adj = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Building distance map'):
        drug = f"D{row['Drug_ID']}"
        targ = f"T{row['Target_ID']}"
        d = float(row['Y'])
        dist[(drug, targ)] = d
        dist[(targ, drug)] = d
        adj.setdefault(drug, set()).add(targ)
        adj.setdefault(targ, set()).add(drug)
    
    # Filter for drugs binding to multiple targets
    multi_drugs = [d for d in adj if d.startswith('D') and len(adj[d]) >= 2]
    total_pairs = comb(len(multi_drugs), 2)
    
    # Prepare CSV for storing violations
    violations = []
    count = 0
    total_bicliques_scanned = 0
    
    # Search for 2x2 bicliques and check rectangular inequality
    for d1, d2 in tqdm(itertools.combinations(multi_drugs, 2), total=total_pairs, desc='Drug pairs'):
        if count >= max_violations:
            break
            
        common_targets = adj[d1].intersection(adj[d2])
        if len(common_targets) < 2:
            continue
            
        for t1, t2 in itertools.combinations(common_targets, 2):
            if count >= max_violations:
                break
            
            total_bicliques_scanned += 1
            
            # … inside your for d1,d2 … for t1,t2 … block …

            # Get individual distances
            d1t1_dist = dist[(d1, t1)]
            d2t2_dist = dist[(d2, t2)]
            d1t2_dist = dist[(d1, t2)]
            d2t1_dist = dist[(d2, t1)]

            # rename for clarity
            a = d1t1_dist
            b = d2t1_dist
            c = d1t2_dist
            d = d2t2_dist

            # build the two intervals [|a-b|, a+b] and [|c-d|, c+d]
            low1, high1 = abs(a - b), a + b
            low2, high2 = abs(c - d), c + d

            # if they *don’t* overlap, that’s a violation
            if max(low1, low2) > min(high1, high2) + tol:
                diff = max(low1, low2) - min(high1, high2)
                violation = {
                    'drug1':   d1,
                    'drug2':   d2,
                    'target1': t1,
                    'target2': t2,
                    'd1t1':    a,
                    'd2t1':    b,
                    'd1t2':    c,
                    'd2t2':    d,
                    'low1':    low1,
                    'high1':   high1,
                    'low2':    low2,
                    'high2':   high2,
                    'difference': diff
                }
                violations.append(violation)
                count += 1

                print(f"Violation {count} for {d1},{d2} vs {t1},{t2}:")
                print(f"  Interval1 = [{low1:.6f}, {high1:.6f}], Interval2 = [{low2:.6f}, {high2:.6f}]")
                print(f"  Gap = {diff:.6f} > tol")


    
    # Save violations to CSV
    violation_percentage = 0
    if violations:
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['drug1', 'drug2', 'target1', 'target2', 
                          'd1t1', 'd2t2', 'd1t2', 'd2t1', 'low1', 'high1', 'low2', 'high2',  'difference']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for v in violations:
                writer.writerow(v)
        
        # Calculate violation percentage
        violation_percentage = (len(violations) / total_bicliques_scanned * 100) if total_bicliques_scanned > 0 else 0
        
        print(f"\nSummary Statistics:")
        print(f"Total bicliques scanned: {total_bicliques_scanned}")
        print(f"Total violations found: {len(violations)}")
        print(f"Violation percentage: {violation_percentage:.2f}%")
        print(f"Saved to {output_path}")
    else:
        print("\nSummary Statistics:")
        print(f"Total bicliques scanned: {total_bicliques_scanned}")
        print(f"Total violations found: 0")
        print(f"Violation percentage: 0.00%")
        print("No rectangular inequality violations found.")
    
    return violations, violation_percentage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find 2x2 bicliques where the rectangular inequality is violated.'
    )
    parser.add_argument('csv_path', help='Path to CSV file with Drug_ID, Target_ID, Y')
    parser.add_argument('--output', default='violations.csv',
                        help='Path to output CSV file (default: violations.csv)')
    parser.add_argument('--max', type=int, default=1000,
                        help='Maximum number of violations to find (default: 100)')
    parser.add_argument('--tol', type=float, default=1e-8,
                        help='Numerical tolerance (default: 1e-8)')
    args = parser.parse_args()

    find_violations(args.csv_path, args.output, args.max, args.tol)