"""Build the evaluation, classical ML, and instance segmentation lessons."""
from pathlib import Path
from notebook_builder import md, code, write_notebook, bootstrap

ROOT = Path(__file__).resolve().parents[1]


def metrics_notebook():
    cells = [
        md(r"""
        # 06 · Measure segmentation and investigate mistakes

        **Level:** intermediate · **Time:** 60–90 minutes · **Runs:** offline, CPU

        **Prerequisites:** NumPy Boolean masks; thresholding and morphology from lessons 01–05.
        You can still run this notebook independently: every required function is defined here.

        **You will learn to:** derive IoU, Dice, precision, and recall; choose an explicit empty-mask
        policy; aggregate multiclass scores; ignore unlabeled pixels; tune a threshold using validation
        images; inspect boundary and class-specific errors.

        Run from the top. Before each plot, predict what you expect. Record one observation after
        each experiment. These tiny synthetic examples teach evaluation mechanics, not real-world accuracy.
        """),
        bootstrap(),
        code("""
        import pandas as pd
        from scipy import ndimage as ndi
        from skimage.segmentation import find_boundaries
        from sklearn.metrics import confusion_matrix, jaccard_score
        from learning.data import make_scene, make_dataset, palette_mask

        image, semantic, _ = make_scene(seed=16, size=96)
        truth = semantic > 0
        # Deliberately imperfect prediction: erosion misses edges; a square is a false positive.
        prediction = ndi.binary_erosion(truth, iterations=2)
        prediction[3:14, 3:17] = True
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, data, title in zip(axes, [image, truth, prediction],
                                  ['Input', 'True foreground', 'Imperfect prediction']):
            ax.imshow(data, cmap='gray'); ax.set_title(title); ax.axis('off')
        plt.tight_layout(); plt.show()
        """),
        md(r"""
        ## Step 1 — Count before using a library

        A **true positive (TP)** is foreground in both masks. **FP** is predicted foreground outside
        the true object. **FN** is missed foreground. **TN** is correctly predicted background.

        \[
        \mathrm{IoU}=\frac{TP}{TP+FP+FN},\quad
        \mathrm{Dice}=\frac{2TP}{2TP+FP+FN},\quad
        \mathrm{precision}=\frac{TP}{TP+FP},\quad
        \mathrm{recall}=\frac{TP}{TP+FN}.
        \]

        IoU penalizes extra and missing regions. Precision asks whether predicted pixels are trustworthy;
        recall asks whether true pixels were found. We use hard Boolean masks, not probability maps.
        """),
        code("""
        def binary_metrics(y_true, y_pred, empty_overlap=1.0):
            y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
            if y_true.shape != y_pred.shape:
                raise ValueError('Masks must have identical shapes.')
            if y_true.size == 0:
                raise ValueError('There must be at least one evaluated pixel.')
            tp = int(np.count_nonzero(y_true & y_pred))
            fp = int(np.count_nonzero(~y_true & y_pred))
            fn = int(np.count_nonzero(y_true & ~y_pred))
            tn = int(np.count_nonzero(~y_true & ~y_pred))
            union = tp + fp + fn
            # Precision/recall use zero when their denominator is zero.
            return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                        IoU=tp / union if union else empty_overlap,
                        Dice=2*tp / (2*tp+fp+fn) if union else empty_overlap,
                        precision=tp / (tp+fp) if tp+fp else 0.0,
                        recall=tp / (tp+fn) if tp+fn else 0.0,
                        accuracy=(tp+tn) / y_true.size)

        result = binary_metrics(truth, prediction)
        print(pd.Series(result).round(4))
        assert np.isclose(result['Dice'], 2*result['IoU']/(1+result['IoU']))
        assert np.isclose(result['IoU'], jaccard_score(truth.ravel(), prediction.ravel()))
        """),
        md("""
        ## Step 2 — Make undefined cases explicit

        If both masks are empty, IoU and Dice contain 0/0. In this lesson their score is **1**:
        correctly predicting absence counts as success. Precision/recall with a zero denominator
        are **0**. Other projects use 0, NaN, or exclude empty cases; report your policy with your results.
        A convention is not a mathematical identity. Never hide the decision by adding an arbitrary epsilon.
        """),
        code("""
        empty = np.zeros((8, 8), bool)
        single = empty.copy(); single[3, 3] = True
        cases = [('both empty', empty, empty), ('false alarm', empty, single),
                 ('complete miss', single, empty), ('perfect', single, single)]
        table = pd.DataFrame([{'case': name, **binary_metrics(t, p)} for name, t, p in cases])
        print(table[['case', 'IoU', 'Dice', 'precision', 'recall']].to_string(index=False))
        print('Alternative policy, both empty:', binary_metrics(empty, empty, empty_overlap=np.nan)['IoU'])
        """),
        md("""
        ## Step 3 — Expose the accuracy trap and locate mistakes

        A background-only prediction can have impressive accuracy when foreground is rare.
        Always report foreground overlap and class-specific scores alongside accuracy.

        In the overlay below: **green = TP, red = FP, blue = FN**. Dark pixels are TN.
        Judge the location of errors as well as their count: losing a thin connection can matter more
        than losing the same number of pixels from a thick interior.
        """),
        code("""
        tiny = np.zeros((100, 100), bool); tiny[40:50, 40:50] = True
        print('Only 1% foreground, predict all background:', binary_metrics(tiny, np.zeros_like(tiny)))
        overlay = np.zeros((*truth.shape, 3), float)
        overlay[truth & prediction] = [0.15, 0.85, 0.25]
        overlay[~truth & prediction] = [1.0, 0.15, 0.1]
        overlay[truth & ~prediction] = [0.1, 0.35, 1.0]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(overlay); ax.set_title('TP green · FP red · FN blue'); ax.axis('off')
        plt.show()
        """),
        md("""
        ## Step 4 — Multiclass confusion and ignored labels

        Each valid pixel has exactly one class: 0 background, 1 disk, 2 rectangle. Rows in our
        confusion matrix are truth; columns are prediction. A value of **255** means annotation
        unavailable, so remove that pixel from both truth and prediction before counting.

        For class *c*: TP is the diagonal entry, FN is the remaining row, FP is the remaining column.
        An ignored pixel is not a fourth class and must not silently become background.
        """),
        code("""
        y_true = semantic.copy()
        y_true[:10, :] = 255
        y_pred = semantic.copy()
        y_pred[semantic == 2] = 1  # deliberately confuse rectangles with disks
        y_pred[5:15, 5:15] = 2

        def multiclass_report(y_true, y_pred, n_classes=3, ignore=255):
            valid = np.asarray(y_true) != ignore
            if not valid.any():
                raise ValueError('No labeled pixels remain.')
            yt, yp = np.asarray(y_true)[valid], np.asarray(y_pred)[valid]
            if not (np.isin(yt, np.arange(n_classes)).all() and
                    np.isin(yp, np.arange(n_classes)).all()):
                raise ValueError('Unexpected class ID in evaluated pixels.')
            cm = confusion_matrix(yt, yp, labels=np.arange(n_classes))
            tp = np.diag(cm).astype(float)
            fn, fp = cm.sum(axis=1)-tp, cm.sum(axis=0)-tp
            union = tp+fp+fn
            # Here absent-in-both classes are NaN, excluded from the macro average.
            iou = np.divide(tp, union, out=np.full(n_classes, np.nan), where=union > 0)
            return cm, pd.DataFrame({'class': np.arange(n_classes), 'support': cm.sum(1),
                                     'TP': tp, 'FP': fp, 'FN': fn, 'IoU': iou})

        cm, class_table = multiclass_report(y_true, y_pred)
        print('Confusion: rows truth / columns prediction'); print(cm)
        print(class_table.round(3).to_string(index=False))
        assert cm.sum() == (y_true != 255).sum()
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].imshow(palette_mask(np.where(y_true == 255, 0, y_true)))
        axes[0].set_title('Truth (top strip ignored)')
        axes[1].imshow(palette_mask(y_pred)); axes[1].set_title('Class confusion')
        for ax in axes: ax.axis('off')
        plt.show()
        """),
        md("""
        ## Step 5 — Specify what you averaged

        **Macro IoU** gives each included class equal weight. **Micro IoU** sums TP/FP/FN over
        classes before dividing, giving frequent classes more influence. Neither is the average of
        image-level scores unless explicitly computed that way. State whether background is included.

        Class-macro averaging and image-macro averaging answer different questions. The example
        below also contrasts equal image weighting with pooling foreground counts across images.
        """),
        code("""
        macro_iou = class_table.IoU.mean()  # pandas skips NaN
        micro_iou = class_table.TP.sum() / (class_table.TP + class_table.FP + class_table.FN).sum()
        print(f'Class macro IoU (background included): {macro_iou:.3f}')
        print(f'Class micro IoU (background included): {micro_iou:.3f}')
        print(f'Foreground-class macro IoU: {class_table.loc[class_table["class"] > 0, "IoU"].mean():.3f}')
        large = np.zeros((40, 40), bool); large[5:35, 5:35] = True
        small = np.zeros((40, 40), bool); small[10:14, 10:14] = True
        per_image = [binary_metrics(large, large), binary_metrics(small, np.zeros_like(small))]
        equal_images = np.mean([r['IoU'] for r in per_image])
        pooled = sum(r['TP'] for r in per_image) / sum(r['TP']+r['FP']+r['FN'] for r in per_image)
        print(f'Equal image mean IoU: {equal_images:.3f}; pooled foreground IoU: {pooled:.3f}')
        """),
        md("""
        ## Step 6 — Tune on validation images, then freeze the decision

        We use a fixed, untrained brightness score to keep this lesson focused on evaluation. It is
        a bounded **score**, not a calibrated probability. Use validation labels to choose a threshold;
        test labels are used only after this choice is frozen. For a learned model, fitting happens on
        a separate training split first. Images—not random pixels—are the independent units here.

        Predict: lowering the threshold should generally increase recall and reduce precision.
        The best threshold depends on your chosen objective; here it is mean per-image foreground IoU.
        """),
        code("""
        val_images, val_masks = make_dataset(n=8, size=64, seed=400)
        test_images, test_masks = make_dataset(n=8, size=64, seed=900)

        def foreground_score(images):
            return np.max(images, axis=-1)

        val_scores = foreground_score(val_images)
        thresholds = np.linspace(0.15, 0.90, 31)
        search_rows = []
        for threshold in thresholds:
            scores = [binary_metrics(t > 0, s >= threshold) for t, s in zip(val_masks, val_scores)]
            search_rows.append({'threshold': threshold,
                                **{key: np.mean([r[key] for r in scores])
                                   for key in ['IoU', 'precision', 'recall']}})
        validation_curve = pd.DataFrame(search_rows)
        best_row = validation_curve.iloc[validation_curve.IoU.argmax()]
        frozen_threshold = float(best_row.threshold)
        print(f'Frozen threshold: {frozen_threshold:.3f}; validation mean IoU: {best_row.IoU:.3f}')
        ax = validation_curve.plot(x='threshold', y=['IoU', 'precision', 'recall'], figsize=(8, 4))
        ax.axvline(frozen_threshold, ls='--', color='black'); ax.set_ylim(0, 1.05)
        ax.set_title('Validation images only'); plt.show()
        """),
        code("""
        # One final test report: do not change the threshold after reading these results.
        test_scores = foreground_score(test_images)
        final_rows = [binary_metrics(t > 0, s >= frozen_threshold)
                      for t, s in zip(test_masks, test_scores)]
        final_report = pd.DataFrame(final_rows)
        print('Frozen test mean per-image metrics:')
        print(final_report[['IoU', 'Dice', 'precision', 'recall']].mean().round(4))
        print('Test image IoUs:', final_report.IoU.round(3).tolist())
        print('Eight synthetic images are too few to establish real-world reliability.')
        """),
        md("""
        ## Step 7 — Look at boundaries

        Area overlap can hide a systematic boundary offset. Below we compare contours and compute
        an illustrative **boundary F1**: a predicted boundary pixel is correct if it is within a chosen
        pixel tolerance of any true boundary pixel; recall reverses that direction. This is not a
        replacement for a benchmark's exact boundary protocol. Tolerance has physical meaning only
        after accounting for image resolution and pixel spacing.
        """),
        code("""
        def boundary_f1(y_true, y_pred, tolerance=2):
            bt = find_boundaries(y_true, mode='inner')
            bp = find_boundaries(y_pred, mode='inner')
            if not bt.any() and not bp.any(): return 1.0
            if not bt.any() or not bp.any(): return 0.0
            near_truth = ndi.distance_transform_edt(~bt) <= tolerance
            near_pred = ndi.distance_transform_edt(~bp) <= tolerance
            precision_b = near_truth[bp].mean()
            recall_b = near_pred[bt].mean()
            return 2*precision_b*recall_b/(precision_b+recall_b) if precision_b+recall_b else 0.0

        print(pd.DataFrame({'tolerance_px': [0, 1, 2, 4],
                            'boundary_F1': [boundary_f1(truth, prediction, t) for t in [0, 1, 2, 4]]}))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(image)
        ax.contour(truth, levels=[0.5], colors=['cyan'], linewidths=1)
        ax.contour(prediction, levels=[0.5], colors=['magenta'], linewidths=1)
        ax.set_title('Truth: cyan · prediction: magenta'); ax.axis('off'); plt.show()
        """),
        md("""
        ## Your experiments

        **Exercise 1 — Hand calculation.** A prediction has TP=8, FP=2, FN=4. Calculate IoU, Dice,
        precision, and recall before checking with Python.

        <details><summary>Worked solution</summary>

        IoU = 8/14 ≈ 0.5714; Dice = 16/22 ≈ 0.7273; precision = 8/10 = 0.8;
        recall = 8/12 ≈ 0.6667. Check that Dice = 2×IoU/(1+IoU).

        </details>

        **Exercise 2 — Change the cost of mistakes.** On the existing validation curve, select the
        threshold with the best mean precision among thresholds with mean recall ≥ 0.95.
        If no threshold qualifies, report that rather than pretending the constraint was met.
        Do not use the current test set to select between objectives.

        <details><summary>Worked solution</summary>

        ```python
        eligible = validation_curve[validation_curve.recall >= 0.95]
        if eligible.empty:
            print('No candidate meets the recall target.')
        else:
            print(eligible.loc[eligible.precision.idxmax()])
        ```

        This changes the decision rule, not the underlying score. For a fresh final comparison,
        predeclare the objective and create a new untouched synthetic test split.

        </details>

        **Exercise 3 — Ignore labels correctly.** Replace all predictions in the ignored strip
        with class 2. Does the confusion matrix change? Explain why.

        <details><summary>Worked solution</summary>

        ```python
        changed = y_pred.copy()
        changed[y_true == 255] = 2
        changed_cm, _ = multiclass_report(y_true, changed)
        assert np.array_equal(cm, changed_cm)
        ```

        Both arrays are filtered by the truth-derived valid mask, so those predictions cannot
        contribute to any evaluated count.

        </details>
        """),
        code("""
        # Experiment workspace: add your calculations and notes here.
        tp, fp, fn = 8, 2, 4
        print({'IoU': tp/(tp+fp+fn), 'Dice': 2*tp/(2*tp+fp+fn),
               'precision': tp/(tp+fp), 'recall': tp/(tp+fn)})
        """),
        md("""
        ## Common mistakes and completion checklist

        - **Dice and IoU disagree unexpectedly:** check the same class, valid pixels, averaging,
          hard threshold, and empty-mask convention were used.
        - **Accuracy looks excellent but objects are missing:** inspect foreground IoU and recall.
        - **Validation keeps improving while test performance disappoints:** repeated tuning can
          overfit validation too. Use more independent data or grouped cross-validation.
        - **Treating every pixel as an independent sample:** uncertainty estimates should usually
          resample independent images, patients, scenes, or acquisition groups, not adjacent pixels.
        - **A single mean hides failure:** report per-class/per-image results and representative errors.

        Before moving on, confirm you can:

        - [ ] Explain every confusion count and derive IoU/Dice.
        - [ ] State an empty-mask and ignored-label policy.
        - [ ] Distinguish class macro, class micro, and image averaging.
        - [ ] Freeze a validation-selected decision before a final test report.
        - [ ] Explain why overlap and boundary measures complement each other.

        **Continue:** lesson 07 learns pixel classifications from examples.

        **Primary references:** [scikit-learn Jaccard/IoU API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html),
        [confusion matrix API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html),
        [data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
        [scikit-image boundary API](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries).
        """),
    ]
    write_notebook(ROOT / 'notebooks/06_metrics_error_analysis.ipynb', cells, '06 Metrics and error analysis')


def classical_notebook():
    cells = [
        md("""
        # 07 · Train a classical pixel classifier

        **Level:** intermediate · **Time:** 75–100 minutes · **Runs:** offline, CPU; small random forests

        **Prerequisites:** array shapes, image filtering, and IoU from lessons 01–06.
        No previously saved notebook output is required.

        **You will learn to:** split whole images before training; transform pixels into feature
        vectors; sample training pixels; compare RGB with multiscale features; select a model on
        validation images; evaluate held-out images; diagnose a controlled domain shift.

        The learner sees colors and local textures, not named objects. Our synthetic labels are
        deliberately correlated with color, so high scores do not demonstrate general shape recognition.
        """),
        bootstrap(),
        code("""
        import pandas as pd
        from scipy import ndimage as ndi
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import confusion_matrix
        from learning.data import make_dataset, palette_mask, show_triplet

        images, masks = make_dataset(n=30, size=64, seed=700, noise=0.08)
        print('Images:', images.shape, images.dtype, 'Masks:', masks.shape, masks.dtype)
        fig, axes = plt.subplots(2, 4, figsize=(12, 5))
        for col, idx in enumerate([0, 4, 8, 12]):
            axes[0, col].imshow(images[idx]); axes[0, col].set_title(f'Image {idx}')
            axes[1, col].imshow(palette_mask(masks[idx]))
        for ax in axes.ravel(): ax.axis('off')
        plt.tight_layout(); plt.show()
        """),
        md("""
        ## Step 1 — Split images before flattening pixels

        Adjacent pixels share texture, illumination, and sometimes almost the same feature vector.
        Randomly splitting all pixels mixes each scene into training and evaluation and gives an
        optimistic estimate for new scenes. We split **image IDs** first, then sample pixels only
        within training images. For repeated patients, videos, or tiles from one slide, split the
        higher-level group instead.

        We reserve 18 training, 6 validation, and 6 test images. These splits are intentionally small
        for speed, not adequate for a production claim.
        """),
        code("""
        ids = np.random.default_rng(42).permutation(len(images))
        train_ids, val_ids, test_ids = ids[:18], ids[18:24], ids[24:]
        assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids))
        print('Train IDs:', train_ids.tolist())
        print('Validation IDs:', val_ids.tolist())
        print('Test IDs: reserved until the model choice is frozen')
        print('Training class fractions:', np.bincount(masks[train_ids].ravel(), minlength=3) / masks[train_ids].size)
        """),
        md("""
        ## Step 2 — Represent one pixel as a feature vector

        The RGB baseline uses three numbers per pixel. The multiscale version adds, for each color
        channel, Gaussian-smoothed intensity, local standard deviation, and gradient magnitude at
        several spatial scales. A larger sigma sees more neighborhood context but can blur small objects.

        These filters have fixed parameters and learn nothing from the dataset. Computing them on a
        validation image is allowed. Any learned normalization, PCA, or feature selection must instead
        be fitted only on training data. We omit coordinates to avoid teaching a fixed-position shortcut.
        """),
        code("""
        def pixel_features(image, multiscale=True, sigmas=(1.0, 3.0)):
            image = np.asarray(image, dtype=np.float32)
            planes = [image[..., c] for c in range(3)]
            names = [f'raw_{name}' for name in 'RGB']
            if multiscale:
                for sigma in sigmas:
                    for c, channel in enumerate('RGB'):
                        a = image[..., c]
                        smooth = ndi.gaussian_filter(a, sigma=sigma, mode='reflect')
                        local_second = ndi.gaussian_filter(a*a, sigma=sigma, mode='reflect')
                        local_std = np.sqrt(np.maximum(local_second-smooth*smooth, 0))
                        gradient = ndi.gaussian_gradient_magnitude(a, sigma=sigma, mode='reflect')
                        planes.extend([smooth, local_std, gradient])
                        names.extend([f'{channel}_mean_s{sigma}', f'{channel}_std_s{sigma}',
                                      f'{channel}_gradient_s{sigma}'])
            features = np.stack(planes, axis=-1).astype(np.float32)
            assert features.shape[:2] == image.shape[:2] and np.isfinite(features).all()
            return features, names

        example_features, feature_names = pixel_features(images[train_ids[0]])
        print('Feature volume (height, width, features):', example_features.shape)
        print('Classifier input (pixels, features):', example_features.reshape(-1, example_features.shape[-1]).shape)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, j in zip(axes, [0, 3, 4, 5]):
            ax.imshow(example_features[..., j], cmap='magma'); ax.set_title(feature_names[j]); ax.axis('off')
        plt.show()
        """),
        md("""
        ## Step 3 — Sample only training pixels

        We cap examples at 250 pixels per class per training image. This reduces memory and makes
        minority classes visible to the learner. Evaluation still uses **every pixel** of each held-out
        image. Balancing the training sample changes the effective class prior; the resulting forest
        probabilities should not automatically be interpreted as calibrated deployment probabilities.

        Trees split on feature thresholds and do not require standardization. If replacing the forest
        with logistic regression or an SVM, use a `Pipeline(StandardScaler(), model)` fitted on the
        sampled training rows; never fit the scaler on validation or test pixels.
        """),
        code("""
        def sampled_training_rows(image_ids, multiscale=True, per_class=250, seed=10):
            rng = np.random.default_rng(seed)
            x_parts, y_parts = [], []
            for idx in image_ids:
                feature_map, names = pixel_features(images[idx], multiscale=multiscale)
                x = feature_map.reshape(-1, feature_map.shape[-1])
                y = masks[idx].ravel()
                selected = []
                for class_id in range(3):
                    available = np.flatnonzero(y == class_id)
                    selected.extend(rng.choice(available, size=min(per_class, len(available)), replace=False))
                selected = np.asarray(selected, dtype=int)
                x_parts.append(x[selected]); y_parts.append(y[selected])
            return np.concatenate(x_parts), np.concatenate(y_parts), names

        x_rgb, y_train, rgb_names = sampled_training_rows(train_ids, multiscale=False)
        x_multi, y_multi, feature_names = sampled_training_rows(train_ids, multiscale=True)
        assert np.array_equal(y_train, y_multi)  # same sampled positions for a fair feature comparison
        print('RGB:', x_rgb.shape, 'Multiscale:', x_multi.shape)
        print('Sampled class counts:', np.bincount(y_train, minlength=3))
        print(f'Multiscale training features: {x_multi.nbytes/1e6:.2f} MB')
        """),
        md("""
        ## Step 4 — Establish a deliberately simple baseline

        First evaluate a background-only predictor. Then train a random forest on RGB. A forest
        averages the predictions of several randomized decision trees. More trees cost computation;
        maximum depth and minimum leaf size constrain the complexity of each tree.

        The metric below pools the confusion counts over the selected images and averages per-class
        IoUs equally, including background. It also reports mean IoU over the two foreground classes.
        """),
        code("""
        def predict_image(model, image, multiscale):
            f, _ = pixel_features(image, multiscale=multiscale)
            return model.predict(f.reshape(-1, f.shape[-1])).reshape(image.shape[:2])

        def evaluate_model(model, image_ids, multiscale):
            cm = np.zeros((3, 3), dtype=np.int64)
            predictions = []
            for idx in image_ids:
                pred = predict_image(model, images[idx], multiscale)
                predictions.append(pred)
                cm += confusion_matrix(masks[idx].ravel(), pred.ravel(), labels=[0, 1, 2])
            tp = np.diag(cm).astype(float)
            union = cm.sum(0)+cm.sum(1)-tp
            ious = np.divide(tp, union, out=np.full(3, np.nan), where=union > 0)
            report = {'background_IoU': ious[0], 'disk_IoU': ious[1], 'rectangle_IoU': ious[2],
                      'macro_IoU': np.nanmean(ious), 'foreground_macro_IoU': np.nanmean(ious[1:])}
            return report, predictions

        dummy = DummyClassifier(strategy='constant', constant=0).fit(x_rgb, y_train)
        rgb_model = RandomForestClassifier(n_estimators=40, max_depth=10,
                                          min_samples_leaf=3, random_state=42, n_jobs=1)
        rgb_model.fit(x_rgb, y_train)
        baseline_report, _ = evaluate_model(dummy, val_ids, False)
        rgb_report, rgb_predictions = evaluate_model(rgb_model, val_ids, False)
        print(pd.DataFrame([baseline_report, rgb_report], index=['background only', 'RGB forest']).round(3))
        """),
        md("""
        ## Step 5 — Compare candidates using validation only

        Add neighborhood features, then change tree depth. Keep the image split, sampled positions,
        random seed, and evaluation metric fixed so the comparison has a clear interpretation.
        Multiscale features are not guaranteed to win: this dataset may be solved mostly by color.

        We choose the highest validation class-macro IoU from three candidates, then freeze the
        candidate. If validation scores tie, the first candidate wins by our declared tie rule.
        """),
        code("""
        candidates = [('RGB depth 10', rgb_model, False, rgb_report)]
        for depth in [6, 12]:
            model = RandomForestClassifier(n_estimators=40, max_depth=depth,
                                           min_samples_leaf=3, random_state=42, n_jobs=1)
            model.fit(x_multi, y_train)
            report, _ = evaluate_model(model, val_ids, True)
            candidates.append((f'Multiscale depth {depth}', model, True, report))
        validation_results = pd.DataFrame([{'candidate': name, **report}
                                           for name, _, _, report in candidates])
        print(validation_results.round(4).to_string(index=False))
        best_index = int(validation_results.macro_IoU.argmax())
        frozen_name, frozen_model, frozen_multiscale, _ = candidates[best_index]
        print('Frozen candidate:', frozen_name)
        """),
        md("""
        ## Step 6 — Evaluate the frozen model once

        The test score estimates performance on new images from this same synthetic generator.
        Do not respond to a disappointing result by choosing another candidate on test scores.
        That would turn the test set into another validation set.
        """),
        code("""
        test_report, test_predictions = evaluate_model(frozen_model, test_ids, frozen_multiscale)
        print('Final held-out report:', frozen_name)
        print(pd.Series(test_report).round(4))
        show_triplet(images[test_ids[0]], masks[test_ids[0]], test_predictions[0],
                     title='Frozen model: held-out image')
        plt.show()
        """),
        md("""
        ## Step 7 — Inspect features and uncertainty cautiously

        A feature-importance chart can reveal reliance on color. The forest's impurity-based importance
        is an internal diagnostic, not proof that a feature causes correct predictions; correlated
        features share importance and the ranking can be biased.

        Predictive entropy is high when the forest distributes probability across several classes.
        Low entropy is not a guarantee of correctness, especially after a distribution shift.
        """),
        code("""
        chosen_names = feature_names if frozen_multiscale else rgb_names
        importance = pd.Series(frozen_model.feature_importances_, index=chosen_names).sort_values()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        importance.tail(10).plot.barh(ax=axes[0]); axes[0].set_title('Forest impurity importance')
        features, _ = pixel_features(images[test_ids[0]], multiscale=frozen_multiscale)
        probabilities = frozen_model.predict_proba(features.reshape(-1, features.shape[-1]))
        entropy = -(probabilities*np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
        entropy = entropy.reshape(images.shape[1:3])
        im = axes[1].imshow(entropy, cmap='magma', vmin=0, vmax=np.log(3))
        axes[1].set_title('Prediction entropy'); axes[1].axis('off'); fig.colorbar(im, ax=axes[1])
        plt.tight_layout(); plt.show()
        """),
        md("""
        ## Step 8 — Make a controlled domain shift

        We reverse the intensity scale of fresh images while keeping their geometry and labels.
        This simulates a deliberately severe change in image appearance. A color-based forest can
        confidently fail. The shift is a diagnostic challenge separate from the original final test;
        it is not evidence of performance on any particular real camera or microscope.

        A remedy needs labeled data representative of the intended use, plus a new independent
        validation/test protocol. Applying a transformation because it improves these challenge labels
        and then reporting the same images as an untouched test would be leakage.
        """),
        code("""
        challenge_images, challenge_masks = make_dataset(n=6, size=64, seed=1700, noise=0.08)
        shifted_images = 1.0-challenge_images

        def evaluate_external(batch, targets):
            cm = np.zeros((3, 3), dtype=np.int64)
            preds = []
            for image, target in zip(batch, targets):
                pred = predict_image(frozen_model, image, frozen_multiscale)
                preds.append(pred)
                cm += confusion_matrix(target.ravel(), pred.ravel(), labels=[0, 1, 2])
            tp = np.diag(cm).astype(float); union = cm.sum(0)+cm.sum(1)-tp
            ious = np.divide(tp, union, out=np.full(3, np.nan), where=union > 0)
            return float(np.nanmean(ious)), preds

        ordinary_iou, _ = evaluate_external(challenge_images, challenge_masks)
        shifted_iou, shifted_predictions = evaluate_external(shifted_images, challenge_masks)
        print(f'Fresh ordinary images: macro IoU {ordinary_iou:.3f}')
        print(f'Same geometry, inverted appearance: macro IoU {shifted_iou:.3f}')
        show_triplet(shifted_images[0], challenge_masks[0], shifted_predictions[0],
                     title='Domain-shift diagnostic')
        plt.show()
        """),
        md("""
        ## Your experiments

        **Exercise 1 — Budget the annotations.** Compare 25, 100, and 250 labeled pixels per class
        per training image. Hold forest settings fixed and report validation macro IoU and the number
        of sampled rows. Leave the final test set out of this experiment.

        <details><summary>Worked solution</summary>

        ```python
        rows = []
        for budget in [25, 100, 250]:
            x, y, _ = sampled_training_rows(train_ids, True, per_class=budget)
            model = RandomForestClassifier(n_estimators=40, max_depth=12,
                min_samples_leaf=3, random_state=42, n_jobs=1).fit(x, y)
            report, _ = evaluate_model(model, val_ids, True)
            rows.append({'pixels_per_class': budget, 'training_rows': len(y), **report})
        print(pd.DataFrame(rows))
        ```

        A larger sample may give only a small gain because nearby pixels are redundant. More
        independent images can add information that more pixels from the same images cannot.

        </details>

        **Exercise 2 — Change the feature scales.** Add sigma=6 to the feature function. Retrain
        a multiscale candidate and compare it on validation. Why might tiny structures suffer?

        <details><summary>Worked solution</summary>

        Change the default to `sigmas=(1.0, 3.0, 6.0)` and rerun feature extraction, sampling, and
        candidate training before making predictions. Each scale adds 9 features. The wider filters
        mix neighborhoods across object boundaries. A model trained with one feature schema cannot
        safely predict using another; names, order, and parameters must match exactly.

        </details>

        **Exercise 3 — Design a leakage-safe adaptation study.** You have labeled images from a
        new camera. Specify the training, validation, and test roles before changing the forest.

        <details><summary>Worked solution</summary>

        Group images by scene/acquisition session and split those groups. Use the training portion
        for fitting, validation for feature choices and augmentation settings, and the untouched new-camera
        test portion for the final report. Keep the previous distribution test as a separate check for
        regression. Do not use the entire new-camera set to fit normalization or choose transformations.

        </details>
        """),
        code("""
        # Your workspace: document the change, prediction, measured validation result, and lesson.
        experiment_log = pd.DataFrame(columns=['change', 'prediction', 'validation_macro_IoU', 'observation'])
        print(experiment_log)
        """),
        md("""
        ## Common mistakes and completion checklist

        - **Suspiciously perfect validation:** check whether images or related groups cross splits.
        - **Prediction shape is wrong:** preserve feature order and reshape using the input height/width.
        - **Model misses a class:** inspect sampled label counts and class-specific IoU.
        - **Smoothing labels:** labels are discrete IDs; filtering features is different from blurring labels.
        - **Interpreting color success as object understanding:** test controlled appearance changes.
        - **Oversized RAM use:** sample training rows, reduce image size or scales, and predict in tiles
          with enough overlap for filters when using large images.

        - [ ] Split independent images/groups before sampling pixels.
        - [ ] Explain each dimension in `(height, width, features)` and `(pixels, features)`.
        - [ ] Compare a simple baseline with at least one richer candidate on validation.
        - [ ] Evaluate every held-out pixel without balancing the evaluation sample.
        - [ ] Freeze the model before the final test and explain its domain-shift limitations.

        **Continue:** lesson 08 introduces learned neural features and tensor layouts.

        **Primary references:** [scikit-image trainable segmentation example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html),
        [RandomForestClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
        [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
        [grouped splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
        """),
    ]
    write_notebook(ROOT / 'notebooks/07_classical_machine_learning.ipynb', cells, '07 Classical machine learning')


def instances_notebook():
    cells = [
        md("""
        # 11 · Instances, panoptic maps, and point prompts

        **Level:** advanced concepts / expert-topic introduction · **Time:** 90–120 minutes
        **Runs:** offline, CPU; no neural checkpoints or downloads

        **Prerequisites:** binary masks, morphology, watershed, IoU, and a basic understanding of
        learned segmentation. All code is self-contained apart from the project bootstrap.

        **You will learn to:** separate touching objects with watershed; match object IDs using IoU
        and one-to-one assignment; distinguish pixel overlap from object detection; encode semantic
        and instance information together; run a point-prompt random-walker experiment.

        These are working classical algorithms and a small panoptic evaluation exercise. They introduce
        ideas used in advanced systems; they do not implement or train SAM, Mask R-CNN, or a foundation model.
        """),
        bootstrap(),
        code("""
        import pandas as pd
        from scipy import ndimage as ndi
        from scipy.optimize import linear_sum_assignment
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed, random_walker, find_boundaries
        from skimage.color import label2rgb

        rng = np.random.default_rng(25)
        height = width = 96
        yy, xx = np.mgrid[:height, :width]
        centers = np.array([[32, 30], [33, 55], [67, 65]])  # row, column
        radius = 18
        squared_distances = np.stack([(yy-y)**2+(xx-x)**2 for y, x in centers])
        foreground_truth = squared_distances.min(axis=0) <= radius**2
        # Overlaps belong to the nearest center, giving non-overlapping visible instance labels.
        true_instances = np.where(foreground_truth, squared_distances.argmin(axis=0)+1, 0).astype(np.int32)
        gray_image = np.clip(0.15+0.65*foreground_truth+rng.normal(0, 0.055, (height, width)), 0, 1)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, data, title in zip(axes, [gray_image, foreground_truth, label2rgb(true_instances, bg_label=0)],
                                  ['Observed image', 'Semantic foreground', 'Three true instances']):
            ax.imshow(data, cmap='gray'); ax.set_title(title); ax.axis('off')
        plt.show()
        """),
        md("""
        ## Step 1 — Semantic versus instance labels

        A semantic map answers *what class is this pixel?* All three disks belong to the same class.
        An instance map answers *which particular object?* IDs 1, 2, and 3 distinguish the disks;
        the integer values themselves have no class meaning. Reassigning IDs does not change the segmentation.

        Connected components merges touching objects. Watershed instead starts from markers and
        assigns regions according to a topographic surface. We use negative distance to background:
        object interiors become basins, and detected distance peaks supply markers.
        """),
        code("""
        observed_foreground = gray_image > 0.5
        connected, n_connected = ndi.label(observed_foreground)
        distance = ndi.distance_transform_edt(observed_foreground)

        def watershed_instances(foreground, min_distance=12):
            dist = ndi.distance_transform_edt(foreground)
            points = peak_local_max(dist, min_distance=min_distance,
                                    labels=foreground.astype(np.uint8), exclude_border=False)
            markers = np.zeros(foreground.shape, np.int32)
            for marker_id, (row, col) in enumerate(points, start=1):
                markers[row, col] = marker_id
            labels = watershed(-dist, markers, mask=foreground)
            return labels.astype(np.int32), markers

        predicted_instances, markers = watershed_instances(observed_foreground)
        print('Connected components:', n_connected, 'Watershed instances:', int(predicted_instances.max()))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].imshow(distance, cmap='magma'); axes[0].set_title('Distance to background')
        rows, cols = np.nonzero(markers)
        axes[0].scatter(cols, rows, c='cyan', marker='+', s=100)
        axes[1].imshow(label2rgb(connected, bg_label=0)); axes[1].set_title('Connected components')
        axes[2].imshow(label2rgb(predicted_instances, bg_label=0)); axes[2].set_title('Watershed instances')
        for ax in axes: ax.axis('off')
        plt.show()
        """),
        md("""
        ## Step 2 — Build an object-to-object IoU table

        Matching raw ID values is wrong: predicted object 1 may correspond to true object 3.
        Instead, compare every true object with every predicted object using mask IoU. Background
        ID 0 is excluded. For multiple semantic classes, matching must also require identical classes.

        A good instance prediction needs both correct object count and accurate individual masks.
        """),
        code("""
        def object_iou_matrix(truth, prediction):
            truth, prediction = np.asarray(truth), np.asarray(prediction)
            if truth.shape != prediction.shape:
                raise ValueError('Instance maps must have equal shapes.')
            true_ids = np.unique(truth); true_ids = true_ids[true_ids > 0]
            pred_ids = np.unique(prediction); pred_ids = pred_ids[pred_ids > 0]
            ious = np.zeros((len(true_ids), len(pred_ids)), dtype=float)
            for i, true_id in enumerate(true_ids):
                a = truth == true_id
                for j, pred_id in enumerate(pred_ids):
                    b = prediction == pred_id
                    ious[i, j] = np.count_nonzero(a & b) / np.count_nonzero(a | b)
            return true_ids, pred_ids, ious

        true_ids, pred_ids, iou_matrix = object_iou_matrix(true_instances, predicted_instances)
        print(pd.DataFrame(iou_matrix, index=[f'true {i}' for i in true_ids],
                           columns=[f'pred {i}' for i in pred_ids]).round(3))
        """),
        md(r"""
        ## Step 3 — One-to-one matching, TP, FP, and FN

        We declare a match only when **IoU > 0.5**. A true object can match at most one prediction
        and vice versa. Unmatched predictions are FP; unmatched true objects are FN. The assignment
        below prioritizes the number of eligible matches, then their total IoU. Gating pairs *before*
        assignment avoids the general error of maximizing unconstrained IoU and only then rejecting pairs.

        SciPy provides a linear assignment solver (often described as Hungarian matching; its current
        implementation is a modified Jonker–Volgenant algorithm). The operation here is one-to-one
        optimal assignment. We report object precision, recall, and F1 at this fixed threshold; this
        is **not average precision (AP)**, which also needs confidence-ranked detections and an evaluation protocol.
        """),
        code("""
        def instance_metrics(truth, prediction, threshold=0.5):
            true_ids, pred_ids, ious = object_iou_matrix(truth, prediction)
            matches = []
            if ious.size:
                eligible = ious > threshold  # strict >0.5 matches the panoptic convention used below
                # One extra eligible match outweighs every possible IoU tie-break gain.
                bonus = min(ious.shape)+1
                reward = eligible.astype(float)*bonus + np.where(eligible, ious, 0)
                row_ids, col_ids = linear_sum_assignment(-reward)
                matches = [(int(true_ids[i]), int(pred_ids[j]), float(ious[i, j]))
                           for i, j in zip(row_ids, col_ids) if eligible[i, j]]
            tp = len(matches); fp = len(pred_ids)-tp; fn = len(true_ids)-tp
            precision = tp/(tp+fp) if tp+fp else (1.0 if not len(true_ids) else 0.0)
            recall = tp/(tp+fn) if tp+fn else 1.0
            detection_denom = tp+0.5*fp+0.5*fn
            rq = tp/detection_denom if detection_denom else 1.0  # object F1 / recognition quality
            sum_iou = sum(m[2] for m in matches)
            sq = sum_iou/tp if tp else np.nan  # no matched-mask quality when TP=0
            pq = sum_iou/detection_denom if detection_denom else np.nan
            return {'TP': tp, 'FP': fp, 'FN': fn, 'precision': precision, 'recall': recall,
                    'object_F1': rq, 'matched_mean_IoU': sq, 'PQ': pq}, matches

        instance_report, matches = instance_metrics(true_instances, predicted_instances)
        print(pd.Series(instance_report).round(3))
        print('Matches (true ID, predicted ID, IoU):', matches)
        # Semantic IDs do not depend on the numeric names of instance IDs.
        renamed = np.where(predicted_instances > 0, predicted_instances+20, 0)
        renamed_report, _ = instance_metrics(true_instances, renamed)
        assert instance_report['object_F1'] == renamed_report['object_F1']
        """),
        md("""
        ## Step 4 — A merged prediction can have perfect foreground IoU

        Replace every foreground instance ID by 1. Its binary foreground mask is unchanged, yet
        multiple objects have merged. Conversely, splitting each disk into several fragments can
        preserve foreground coverage while producing extra detections. Inspect both area and instance scores.

        **Empty-set policy:** when both instance sets are empty we assign object F1=1, but PQ and
        matched-mean-IoU are NaN because no segment class was evaluated. With objects on only one side,
        object F1 and PQ are 0. Report this convention; benchmark policies may differ.
        """),
        code("""
        merged = (true_instances > 0).astype(np.int32)
        split = np.zeros_like(true_instances)
        for object_id, (_, center_x) in enumerate(centers, start=1):
            region = true_instances == object_id
            split[region & (xx < center_x)] = 2*object_id-1
            split[region & (xx >= center_x)] = 2*object_id

        rows = []
        for name, pred in [('perfect', true_instances), ('watershed', predicted_instances),
                           ('all objects merged', merged), ('each object split', split)]:
            binary_union = (true_instances > 0) | (pred > 0)
            foreground_iou = np.count_nonzero((true_instances > 0) & (pred > 0))/binary_union.sum()
            report, _ = instance_metrics(true_instances, pred)
            rows.append({'case': name, 'foreground_IoU': foreground_iou, **report})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        empty = np.zeros_like(true_instances)
        assert instance_metrics(empty, empty)[0]['object_F1'] == 1.0
        assert instance_metrics(true_instances, empty)[0]['FN'] == 3
        """),
        md("""
        ## Step 5 — Experiment with marker spacing

        Marker spacing is an object-scale assumption. Too few markers merge objects; too many
        can split one object into several. Before running the sweep, predict what a large
        `min_distance` will do. On a real project, select this parameter on validation scenes,
        then assess a separate test set. This single-scene sweep is an illustration, not a generalization result.
        """),
        code("""
        rows = []
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, spacing in zip(axes, [5, 12, 28]):
            pred, _ = watershed_instances(observed_foreground, min_distance=spacing)
            report, _ = instance_metrics(true_instances, pred)
            rows.append({'min_distance': spacing, 'objects': len(np.unique(pred))-1, **report})
            ax.imshow(label2rgb(pred, bg_label=0)); ax.set_title(f'min_distance={spacing}'); ax.axis('off')
        plt.show()
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        """),
        md(r"""
        ## Step 6 — Combine semantic and instance labels: a panoptic map

        **Things** are countable objects (our disks); **stuff** is a region category (our background).
        Each pixel gets both a semantic class and an instance/segment identity. Here semantic class 1
        is background stuff; class 2 is a disk thing. Stuff uses instance ID 0; disks use positive IDs.
        We pack them as `semantic * 1000 + instance`. This is our local teaching format, not COCO's file format.

        Panoptic Quality for one class is
        \(PQ=\sum_{(p,g)\in TP}IoU(p,g)/(TP+0.5FP+0.5FN)\).
        When TP>0, it factors into matched-mask quality SQ and recognition quality RQ.
        We match only within a semantic class and then average class PQ values equally.
        Official evaluation additionally defines void/crowd handling, dataset aggregation, and class inclusion.
        """),
        code("""
        def pack_panoptic(instance_map):
            if instance_map.min() < 0 or instance_map.max() >= 1000:
                raise ValueError('This teaching encoding requires instance IDs in [0, 999].')
            semantic_map = np.where(instance_map > 0, 2, 1).astype(np.int32)
            packed = semantic_map*1000 + instance_map
            return semantic_map, packed

        true_semantic, true_panoptic = pack_panoptic(true_instances)
        pred_semantic, pred_panoptic = pack_panoptic(predicted_instances)
        assert np.array_equal(pred_panoptic//1000, pred_semantic)
        assert np.array_equal(pred_panoptic%1000, predicted_instances)
        print('Segment IDs:', np.unique(pred_panoptic).tolist())
        segments = [{'segment_id': int(s), 'semantic_class': int(s//1000),
                     'instance_id': int(s%1000), 'area': int((pred_panoptic == s).sum())}
                    for s in np.unique(pred_panoptic)]
        print(pd.DataFrame(segments).to_string(index=False))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].imshow(pred_semantic, cmap='viridis', vmin=1, vmax=2); axes[0].set_title('Semantic classes')
        axes[1].imshow(label2rgb(pred_panoptic, bg_label=1000)); axes[1].set_title('Panoptic segments')
        for ax in axes: ax.axis('off')
        plt.show()
        """),
        code("""
        # Same-class matching: treat all pixels of the stuff class as one segment.
        class_rows = []
        for class_id, name in [(1, 'background stuff'), (2, 'disk things')]:
            if class_id == 1:
                truth_for_class = (true_semantic == class_id).astype(np.int32)
                pred_for_class = (pred_semantic == class_id).astype(np.int32)
            else:
                truth_for_class = np.where(true_semantic == class_id, true_instances, 0)
                pred_for_class = np.where(pred_semantic == class_id, predicted_instances, 0)
            report, _ = instance_metrics(truth_for_class, pred_for_class, threshold=0.5)
            class_rows.append({'class': name, **report})
        panoptic_report = pd.DataFrame(class_rows)
        print(panoptic_report.round(4).to_string(index=False))
        print('Toy class-mean PQ:', round(panoptic_report.PQ.mean(), 4))
        print('No void/crowd regions in this synthetic scene; this is not a benchmark evaluator.')
        """),
        md("""
        ## Step 7 — Give point prompts to a classical algorithm

        A prompt supplies information about the desired mask. We place positive foreground and
        negative/background point labels on a small image. In random walker, label 0 means unknown,
        1 means background seed, and 2 means foreground seed. These positive integers are categories;
        a negative numeric label has a different API meaning and is not a negative prompt.

        The algorithm propagates seed labels using image intensity differences. `beta` controls
        sensitivity to those differences. The brute-force linear solver is convenient for this 64×64
        example; large images require more suitable solvers and memory planning. This is a graph-based
        method with no learned image representation.
        """),
        code("""
        rr, cc = np.mgrid[:64, :64]
        prompt_truth = ((rr-32)/16)**2 + ((cc-32)/20)**2 <= 1
        prompt_image = np.clip(0.2+0.55*prompt_truth+
                               np.random.default_rng(8).normal(0, 0.045, (64, 64)), 0, 1)
        prompt_seeds = np.zeros((64, 64), dtype=np.int32)
        background_points = [(5, 5), (5, 58), (58, 5), (58, 58)]
        foreground_points = [(32, 32)]
        for row, col in background_points: prompt_seeds[row, col] = 1
        for row, col in foreground_points: prompt_seeds[row, col] = 2
        prompted_labels = random_walker(prompt_image, prompt_seeds, beta=90, mode='bf', channel_axis=None)
        prompted_mask = prompted_labels == 2

        def mask_iou(a, b):
            union = np.count_nonzero(a | b)
            return np.count_nonzero(a & b)/union if union else 1.0

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].imshow(prompt_image, cmap='gray')
        for points, color, marker in [(background_points, 'red', 'x'), (foreground_points, 'lime', '+')]:
            rows, cols = np.array(points).T
            axes[0].scatter(cols, rows, c=color, marker=marker, s=70)
        axes[0].set_title('Red: background · green: foreground')
        axes[1].imshow(prompt_truth, cmap='gray'); axes[1].set_title('Truth')
        axes[2].imshow(prompted_mask, cmap='gray'); axes[2].set_title(f'Random walker: IoU={mask_iou(prompt_truth, prompted_mask):.3f}')
        for ax in axes: ax.axis('off')
        plt.show()
        """),
        md("""
        ## Step 8 — Prompt placement is part of the experiment

        Move a foreground prompt into the background and observe the result. Seed constraints are
        treated as known labels, so an incorrect prompt can force a wrong answer even with good
        image contrast. For a fair comparison of promptable systems, define how prompts are chosen,
        how many corrective clicks are allowed, and when the interaction stops.

        Ground-truth-aware prompt placement, used here for teaching, is not an automatic deployment pipeline.
        """),
        code("""
        wrong_seeds = prompt_seeds.copy()
        wrong_seeds[32, 32] = 0
        wrong_seeds[8, 32] = 2  # intentionally incorrect foreground prompt
        wrong_mask = random_walker(prompt_image, wrong_seeds, beta=90,
                                   mode='bf', channel_axis=None) == 2
        print(f'Correct center prompt IoU: {mask_iou(prompt_truth, prompted_mask):.3f}')
        print(f'Misplaced foreground prompt IoU: {mask_iou(prompt_truth, wrong_mask):.3f}')
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].imshow(prompted_mask, cmap='gray'); axes[0].set_title('Correct foreground prompt')
        axes[1].imshow(wrong_mask, cmap='gray'); axes[1].set_title('Misplaced foreground prompt')
        for ax in axes: ax.axis('off')
        plt.show()
        """),
        md("""
        ## Bridge to learned instance and promptable systems

        | Method | Main input beyond the image | Output idea | What to investigate next |
        |---|---|---|---|
        | Watershed | Foreground mask and markers | Non-overlapping region IDs | Marker quality and touching-object errors |
        | Random walker | Labeled seed pixels | Dense labels propagated from seeds | Prompt budget and contrast sensitivity |
        | Mask R-CNN | Trained weights; labels during training | Detected objects, class scores, boxes, and masks | Detection confidence, overlap handling, class-aware AP |
        | SAM | Pretrained weights and prompts such as points/boxes | Prompt-conditioned masks | Ambiguous masks, prompt protocol, domain suitability |
        | Panoptic system | A model and semantic category definitions | A class and segment identity for every pixel | Thing/stuff reconciliation and official PQ evaluation |

        A foundation model has learned from a large pretraining dataset. Our random walker has not.
        A mask generator also does not automatically assign your dataset's semantic categories or
        resolve all overlapping proposals into a panoptic map. Model-specific work needs its own
        dependencies, weights, device budget, and data evaluation; no external model is downloaded here.

        Follow the [official SAM repository](https://github.com/facebookresearch/segment-anything) for its
        original model and prompt examples, and the [Torchvision Mask R-CNN documentation](https://docs.pytorch.org/vision/stable/models/mask_rcnn.html)
        for available model interfaces. These links are optional extensions, not prerequisites to run this lesson.
        """),
        md("""
        ## Your experiments

        **Exercise 1 — Rename every object.** Change predicted IDs by an arbitrary permutation.
        Prove that instance precision and recall remain unchanged.

        <details><summary>Worked solution</summary>

        ```python
        permuted = np.zeros_like(predicted_instances)
        ids = np.unique(predicted_instances)
        ids = ids[ids > 0]
        for old, new in zip(ids, ids[::-1] + 100):
            permuted[predicted_instances == old] = new
        changed, _ = instance_metrics(true_instances, permuted)
        assert changed['precision'] == instance_report['precision']
        assert changed['recall'] == instance_report['recall']
        ```

        The IoU matrix columns are permuted, but geometry and eligible one-to-one matches are unchanged.

        </details>

        **Exercise 2 — Raise the IoU threshold.** Compare thresholds 0.5, 0.75, and 0.9. Explain why
        object recall may decrease although the predicted mask did not change. This is a fixed-threshold
        score sweep; do not call its average COCO AP.

        <details><summary>Worked solution</summary>

        ```python
        print(pd.DataFrame([{'threshold': t,
            **instance_metrics(true_instances, predicted_instances, threshold=t)[0]}
            for t in [0.5, 0.75, 0.9]]))
        ```

        Stricter thresholds reject less accurate geometric matches, turning one previous TP into an
        FP and an FN. Results may remain unchanged if all matched masks already have very high IoU.

        </details>

        **Exercise 3 — Repair a prompt.** Keep the misplaced foreground prompt and add a correct
        foreground prompt at the center. Then instead remove the wrong prompt. Compare the two outcomes.

        <details><summary>Worked solution</summary>

        ```python
        added = wrong_seeds.copy(); added[32, 32] = 2
        removed = added.copy(); removed[8, 32] = 0
        for name, seeds in [('added only', added), ('removed wrong seed', removed)]:
            result = random_walker(prompt_image, seeds, beta=90, mode='bf', channel_axis=None) == 2
            print(name, mask_iou(prompt_truth, result))
        ```

        Adding a correct seed does not erase the contradictory hard constraint. Removing the wrong
        seed restores a consistent set of labels. Always inspect the prompts as well as the output.

        </details>
        """),
        code("""
        # Your workspace: record counts as well as overlap.
        trial_rows = [{'IoU_threshold': t, **instance_metrics(true_instances, predicted_instances, t)[0]}
                      for t in [0.5, 0.75, 0.9]]
        print(pd.DataFrame(trial_rows).round(3).to_string(index=False))
        """),
        md("""
        ## Common mistakes and completion checklist

        - **Comparing ID numbers directly:** match masks by geometry and semantic class.
        - **Counting a merge as several true positives:** each prediction gets at most one match.
        - **Reporting foreground IoU as instance quality:** splits/merges can preserve the foreground union.
        - **Calling object F1 AP:** AP needs confidence-ranked detections and a complete benchmark protocol.
        - **Calling a packed integer image a panoptic model:** encoding organizes outputs; it does not learn them.
        - **Using label -1 as a background prompt in random walker:** positive labels identify seed classes;
          check the API's separate inactive-pixel meaning for negative labels.

        - [ ] Explain why connected components merges touching disks.
        - [ ] Match instances without relying on their numeric IDs.
        - [ ] Count TP/FP/FN for split and merged masks and state the IoU threshold.
        - [ ] Encode and decode semantic/instance IDs and compute the toy classwise PQ.
        - [ ] Explain the difference between classical seed propagation and a pretrained promptable model.
        - [ ] State what this toy evaluation omits before adapting it to real data.

        **Continue:** use these evaluation ideas in the capstone, with a declared task and independent data splits.

        **Primary references:** [scikit-image segmentation API (watershed and random walker)](https://scikit-image.org/docs/stable/api/skimage.segmentation.html),
        [SciPy assignment API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html),
        [official COCO panoptic evaluation repository](https://github.com/cocodataset/panopticapi),
        [official SAM repository](https://github.com/facebookresearch/segment-anything),
        [Torchvision Mask R-CNN](https://docs.pytorch.org/vision/stable/models/mask_rcnn.html).
        """),
    ]
    write_notebook(ROOT / 'notebooks/11_instance_panoptic_prompting.ipynb', cells, '11 Instances, panoptic maps, and prompts')


if __name__ == '__main__':
    metrics_notebook()
    classical_notebook()
    instances_notebook()
