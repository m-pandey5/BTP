"""
Exact ML-Assisted Benders Decomposition for the p-Median Problem.

Source: bd_ml_CODE.pdf
"""

import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ExactMLBendersPMedian:
    def __init__(self, distance_matrix, p, ml_start_iter=4, max_iter=100, tol=1e-8):
        self.d = np.asarray(distance_matrix, dtype=float)
        self.n_customers, self.n_facilities = self.d.shape
        self.p = p
        self.ml_start_iter = ml_start_iter
        self.max_iter = max_iter
        self.tol = tol

        self.cuts = []
        self.history = []
        self.best_y = None
        self.best_cost = float("inf")

        self.X_train = []
        self.y_train = []
        self.model = None

        self.all_combinations = list(
            itertools.combinations(range(self.n_facilities), self.p)
        )

    def y_from_combination(self, comb):
        y = np.zeros(self.n_facilities, dtype=int)
        y[list(comb)] = 1
        return y

    def true_assignment_costs(self, y):
        open_facilities = np.where(y == 1)[0]
        return np.min(self.d[:, open_facilities], axis=1)

    def cut_rhs(self, cut, y):
        customer, alpha, coeffs = cut
        return alpha + sum(coeffs[j] * y[j] for j in coeffs)

    def master_value_for_y(self, y):
        theta = np.zeros(self.n_customers)
        for i in range(self.n_customers):
            rhs_values = [0.0]
            for cut in self.cuts:
                if cut[0] == i:
                    rhs_values.append(self.cut_rhs(cut, y))
            theta[i] = max(rhs_values)
        return float(np.sum(theta)), theta

    def solve_master_exact(self, probs=None):
        best_obj = float("inf")
        best_y = None
        best_theta = None
        best_ml_score = -float("inf")

        for comb in self.all_combinations:
            y = self.y_from_combination(comb)
            obj, theta = self.master_value_for_y(y)

            ml_score = 0.0 if probs is None else float(np.sum(probs[list(comb)]))

            if obj < best_obj - self.tol:
                best_obj = obj
                best_y = y
                best_theta = theta
                best_ml_score = ml_score
            elif abs(obj - best_obj) <= self.tol and ml_score > best_ml_score:
                best_y = y
                best_theta = theta
                best_ml_score = ml_score

        return best_y, best_theta, best_obj

    def generate_cut(self, customer, y):
        open_facilities = np.where(y == 1)[0]
        D = float(np.min(self.d[customer, open_facilities]))

        coeffs = {}
        for j in range(self.n_facilities):
            if self.d[customer, j] < D - self.tol:
                coeffs[j] = -(D - float(self.d[customer, j]))

        return customer, D, coeffs

    def add_violated_cuts(self, y, theta, true_costs):
        added = 0
        for i in range(self.n_customers):
            if theta[i] + self.tol < true_costs[i]:
                self.cuts.append(self.generate_cut(i, y))
                added += 1
        return added

    def build_features(self, current_y):
        avg_dist = np.mean(self.d, axis=0)
        min_dist = np.min(self.d, axis=0)
        max_dist = np.max(self.d, axis=0)
        std_dist = np.std(self.d, axis=0)

        freq = np.zeros(self.n_facilities)
        avg_obj_when_selected = np.zeros(self.n_facilities)
        count_selected = np.zeros(self.n_facilities)

        for rec in self.history:
            y = rec["y"]
            cost = rec["true_cost"]
            freq += y
            avg_obj_when_selected += y * cost
            count_selected += y

        if len(self.history) > 0:
            freq = freq / len(self.history)

        mask = count_selected > 0
        avg_obj_when_selected[mask] /= count_selected[mask]
        avg_obj_when_selected[~mask] = (
            self.best_cost if np.isfinite(self.best_cost) else 0.0
        )

        incumbent = (
            self.best_y if self.best_y is not None
            else np.zeros(self.n_facilities)
        )

        return np.column_stack([
            avg_dist,
            min_dist,
            max_dist,
            std_dist,
            current_y,
            freq,
            incumbent,
            avg_obj_when_selected,
        ])

    def update_training_data(self, current_y):
        if self.best_y is None:
            return

        X = self.build_features(current_y)
        labels = self.best_y.astype(int)

        for j in range(self.n_facilities):
            self.X_train.append(X[j].tolist())
            self.y_train.append(int(labels[j]))

    def train_model(self):
        if len(self.y_train) < 2 or len(set(self.y_train)) < 2:
            return

        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.model.fit(np.asarray(self.X_train), np.asarray(self.y_train))

    def predict_probabilities(self, current_y):
        if self.model is None:
            return None

        X = self.build_features(current_y)
        return self.model.predict_proba(X)[:, 1]

    def solve(self):
        current_y = np.zeros(self.n_facilities, dtype=int)
        probs = None

        for it in range(1, self.max_iter + 1):
            if it >= self.ml_start_iter:
                self.train_model()
                probs = self.predict_probabilities(current_y)

            y, theta, master_obj = self.solve_master_exact(probs=probs)

            true_costs = self.true_assignment_costs(y)
            true_cost = float(np.sum(true_costs))

            if true_cost < self.best_cost - self.tol:
                self.best_cost = true_cost
                self.best_y = y.copy()

            added = self.add_violated_cuts(y, theta, true_costs)

            self.history.append({
                "iteration": it,
                "y": y.copy(),
                "master_obj": master_obj,
                "true_cost": true_cost,
                "best_cost": self.best_cost,
                "cuts_added": added,
            })

            self.update_training_data(y)
            current_y = y.copy()

            print(
                f"Iter {it:02d} | "
                f"LB (master)={master_obj:7.3f} | "
                f"True={true_cost:7.3f} | "
                f"Best={self.best_cost:7.3f} | "
                f"Cuts added={added:2d} | "
                f"y={y.tolist()}"
            )

            if added == 0:
                print("\nConverged: no violated Benders cuts.")
                break

        return self.best_y, self.best_cost


if __name__ == "__main__":
    distance_matrix = np.array([
        [2, 6, 8, 9],
        [3, 5, 7, 8],
        [6, 2, 4, 7],
        [7, 3, 2, 4],
        [8, 6, 3, 2],
    ], dtype=float)

    p = 2

    solver = ExactMLBendersPMedian(
        distance_matrix=distance_matrix,
        p=p,
        ml_start_iter=3,
        max_iter=50,
    )

    best_y, best_cost = solver.solve()

    print("\nFinal Result")
    print("Selected facilities:", np.where(best_y == 1)[0].tolist())
    print("Best objective:", best_cost)
