import sys
import os
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, QShortcut
from PySide6.QtGui import QPalette, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QPushButton, QLabel, QSlider, QGroupBox, QCheckBox,
    QFileDialog, QScrollArea, QFormLayout, QProgressBar
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

from sklearn.datasets import make_blobs, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Optional: Modern AI Sentence Embeddings
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# Optional: UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class HOGTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to extract HOG features from flattened image arrays."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from skimage.feature import hog
            n_features = X.shape[1]
            side = int(np.sqrt(n_features))
            if side * side != n_features:
                raise ValueError(
                    f"HOG requires square-flattened images. Got {n_features} features, "
                    f"which is not a perfect square."
                )
            img_shape = (side, side)
            return np.array([hog(x.reshape(img_shape), pixels_per_cell=(4, 4)) for x in X])
        except ImportError:
            print("Warning: scikit-image not installed. Skipping HOG transform.")
            return X


class PipelineWorker(QThread):
    """Runs the entire data pipeline in a background thread."""
    finished = Signal(object, object, object, object)  # X_reduced, y_labels, legend_info, meta_data
    error = Signal(str)
    progress = Signal(str, int)

    _st_model = None

    def __init__(self, data, data_type, embed_method, reduce_method, params):
        super().__init__()
        self.data = data
        self.data_type = data_type
        self.embed_method = embed_method
        self.reduce_method = reduce_method
        self.params = params

    def run(self):
        try:
            X = None
            y = self.params.get('labels', None)
            meta = {}

            if self.data_type == "Raw Text":
                self.progress.emit("Embedding text...", 20)
                
                lower = self.params.get('lowercase', True)
                stop_words = 'english' if self.params.get('remove_stopwords', True) and lower else None
                
                if self.embed_method == "Sentence Transformer (AI)":
                    if not ST_AVAILABLE:
                        raise ImportError("sentence-transformers not installed.")
                    self.progress.emit("Loading AI Model (may download on first run)...", 40)
                    
                    if PipelineWorker._st_model is None:
                        PipelineWorker._st_model = SentenceTransformer('all-MiniLM-L6-v2')
                        
                    X = PipelineWorker._st_model.encode(self.data, show_progress_bar=False)
                else:
                    if self.embed_method == "TF-IDF":
                        vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words, lowercase=lower)
                    elif self.embed_method == "Bag of Words":
                        vectorizer = CountVectorizer(max_features=1000, stop_words=stop_words, lowercase=lower)
                    elif self.embed_method == "Hashing":
                        vectorizer = HashingVectorizer(n_features=1000, alternate_sign=False, stop_words=stop_words, lowercase=lower)
                    else:
                        raise ValueError("Unknown embedding method")
                    X = vectorizer.fit_transform(self.data).toarray()
            
            elif self.data_type == "Numeric (Tabular)":
                self.progress.emit("Processing numeric data...", 20)
                X = self.data
                if self.embed_method == "HOG (Image Features)":
                    transformer = HOGTransformer()
                    X = transformer.fit_transform(X)
            
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = np.nan_to_num(X.astype(np.float32))

            self.progress.emit(f"Computing {self.reduce_method} projection...", 60)
            n_components = 3 if self.params.get('is_3d', False) else 2
            X_reduced = None

            if self.reduce_method == "PCA":
                reducer = PCA(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X)
                meta['explained_variance'] = reducer.explained_variance_ratio_
            elif self.reduce_method == "t-SNE":
                perplexity = min(self.params.get('perplexity', 30), len(X) - 1)
                init_method = 'pca' if len(X) >= 150 else 'random'
                reducer = TSNE(n_components=n_components, perplexity=perplexity, init=init_method, learning_rate='auto', random_state=42)
                X_reduced = reducer.fit_transform(X)
            elif self.reduce_method == "UMAP" and UMAP_AVAILABLE:
                n_neighbors = self.params.get('n_neighbors', 15)
                reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
                X_reduced = reducer.fit_transform(X)
            elif self.reduce_method == "Isomap":
                n_neighbors = self.params.get('n_neighbors', 15)
                reducer = Isomap(n_components=n_components, n_neighbors=n_neighbors)
                X_reduced = reducer.fit_transform(X)

            if y is None or len(y) == 0:
                y = np.zeros(len(X_reduced))

            self.progress.emit("Rendering plot...", 90)
            self.finished.emit(X_reduced, y, self.params.get('legend_info', None), meta)
        except Exception as e:
            self.error.emit(str(e))


class UltimateEmbeddingVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Data & Embedding Pipeline Visualizer")
        self.resize(1280, 860)
        self.setStyleSheet("background-color: #2b2b2b; color: #e0e0e0;")

        self.current_data_type = "Built-in: Text (AI vs Nature)"
        self.current_embed_method = "TF-IDF"
        self.current_reduce_method = "PCA"
        self.is_3d = False
        
        self.raw_data = None
        self.data_to_process = None
        self.labels = None
        self.legend_info = None
        
        self.current_X_reduced = None
        self.current_y_labels = None
        self.worker = None
        
        self._load_builtin_data()
        self._init_ui()
        self._update_plot()

    def _load_builtin_data(self):
        self.builtin_data = {}
        ai_texts = ["Artificial intelligence is transforming the world.", "Neural networks can learn complex patterns.", "Machine learning models require large datasets.", "Deep learning is a subset of AI.", "Robots are using computer vision to navigate."] * 10
        nature_texts = ["The forest is full of tall green trees.", "Rivers flow down the rocky mountains.", "Birds sing beautifully in the morning.", "The ocean waves crash against the shore.", "Wildlife thrives in the dense jungle."] * 10
        texts = ai_texts + nature_texts
        labels = np.array([0]*len(ai_texts) + [1]*len(nature_texts))
        self.builtin_data["Built-in: Text (AI vs Nature)"] = ("Raw Text", texts, labels, {0: "AI", 1: "Nature"})

        digits = load_digits()
        self.builtin_data["Built-in: Images (Digits 8x8)"] = ("Numeric (Tabular)", digits.data, digits.target, {i: str(i) for i in range(10)})

        X_blob, y_blob = make_blobs(n_samples=300, n_features=50, centers=5, cluster_std=2.0, random_state=42)
        self.builtin_data["Built-in: Numeric Blobs"] = ("Numeric (Tabular)", X_blob, y_blob, {i: f"Cluster {i}" for i in range(5)})

    def _group_style(self):
        return """
            QGroupBox { font-weight: bold; border: 1px solid #555555; border-radius: 4px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._update_plot)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setSpacing(15)
        
        self._create_control_panel(control_layout)
        scroll.setWidget(control_container)
        main_layout.addWidget(scroll, 1)

        self.figure = Figure(figsize=(10, 8), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: transparent;")
        main_layout.addWidget(self.canvas, 4)

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        self.statusBar().showMessage("Ready. Press Ctrl+R to run.")
        self.statusBar().setStyleSheet("background-color: #1e1e1e; color: #aaaaaa;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(200, 16)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _create_control_panel(self, layout):
        # 1. Data Source
        group_data = QGroupBox("1. Data Source")
        group_data.setStyleSheet(self._group_style())
        data_layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "Built-in: Text (AI vs Nature)", 
            "Built-in: Images (Digits 8x8)", 
            "Built-in: Numeric Blobs",
            "Upload: CSV/JSON (Text)", 
            "Upload: CSV/JSON (Numeric Vectors)",
            "Upload: TXT (Line-by-Line Text)"
        ])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        data_layout.addWidget(self.source_combo)
        
        self.upload_btn = QPushButton("📁 Select File")
        self.upload_btn.setEnabled(False)
        self.upload_btn.clicked.connect(self._on_upload_file)
        data_layout.addWidget(self.upload_btn)
        
        self.col_form = QFormLayout()
        self.text_col_combo = QComboBox()
        self.text_col_combo.currentIndexChanged.connect(self._on_col_changed)
        self.label_col_combo = QComboBox()
        self.label_col_combo.currentIndexChanged.connect(self._on_col_changed)
        self.col_form.addRow("Feature Col:", self.text_col_combo)
        self.col_form.addRow("Label Col:", self.label_col_combo)
        data_layout.addLayout(self.col_form)
        
        group_data.setLayout(data_layout)
        layout.addWidget(group_data)

        # 2. Preprocessing & Embedding
        self.group_embed = QGroupBox("2. Embedding & Preprocessing")
        self.group_embed.setStyleSheet(self._group_style())
        embed_layout = QVBoxLayout()
        
        pp_layout = QHBoxLayout()
        self.cb_lower = QCheckBox("Lowercase")
        self.cb_lower.setChecked(True)
        self.cb_stop = QCheckBox("Remove Stopwords")
        self.cb_stop.setChecked(True)
        pp_layout.addWidget(self.cb_lower)
        pp_layout.addWidget(self.cb_stop)
        embed_layout.addLayout(pp_layout)

        self.embed_combo = QComboBox()
        self.embed_combo.addItems(["TF-IDF", "Bag of Words", "Hashing"])
        if ST_AVAILABLE:
            self.embed_combo.addItem("Sentence Transformer (AI)")
        self.embed_combo.addItem("HOG (Image Features)")
        self.embed_combo.currentTextChanged.connect(self._on_embed_method_changed)
        embed_layout.addWidget(self.embed_combo)
        
        self.group_embed.setLayout(embed_layout)
        layout.addWidget(self.group_embed)

        # 3. Dimensionality Reduction
        group_reduce = QGroupBox("3. Dimensionality Reduction")
        group_reduce.setStyleSheet(self._group_style())
        reduce_layout = QVBoxLayout()
        self.reduce_combo = QComboBox()
        methods = ["PCA", "t-SNE", "Isomap"]
        if UMAP_AVAILABLE:
            methods.append("UMAP")
        self.reduce_combo.addItems(methods)
        self.reduce_combo.currentTextChanged.connect(self._on_reduce_method_changed)
        reduce_layout.addWidget(self.reduce_combo)
        group_reduce.setLayout(reduce_layout)
        layout.addWidget(group_reduce)

        # 4. Parameters
        group_params = QGroupBox("4. Algorithm Parameters")
        group_params.setStyleSheet(self._group_style())
        params_layout = QVBoxLayout()
        self.perp_label = QLabel("t-SNE Perplexity: 30")
        self.perplexity_slider = QSlider(Qt.Horizontal)
        self.perplexity_slider.setRange(5, 100)
        self.perplexity_slider.setValue(30)
        self.perplexity_slider.valueChanged.connect(self._on_perplexity_changed)
        params_layout.addWidget(self.perp_label)
        params_layout.addWidget(self.perplexity_slider)

        self.neigh_label = QLabel("Neighbors (UMAP/Isomap): 15")
        self.neigh_slider = QSlider(Qt.Horizontal)
        self.neigh_slider.setRange(5, 100)
        self.neigh_slider.setValue(15)
        self.neigh_slider.valueChanged.connect(self._on_neighbors_changed)
        params_layout.addWidget(self.neigh_label)
        params_layout.addWidget(self.neigh_slider)
        group_params.setLayout(params_layout)
        layout.addWidget(group_params)

        # 5. Visualization
        group_visual = QGroupBox("5. Visualization Layers")
        group_visual.setStyleSheet(self._group_style())
        visual_layout = QVBoxLayout()
        
        self.cb_3d = QCheckBox("Render in 3D")
        self.cb_3d.stateChanged.connect(self._on_3d_toggle)
        visual_layout.addWidget(self.cb_3d)

        # New Viz Checkboxes
        viz_flags_layout = QHBoxLayout()
        self.cb_hull = QCheckBox("Convex Hulls")
        self.cb_hull.setChecked(True)
        self.cb_hull.stateChanged.connect(self._update_plot)
        self.cb_density = QCheckBox("Density Contours")
        self.cb_density.stateChanged.connect(self._update_plot)
        viz_flags_layout.addWidget(self.cb_hull)
        viz_flags_layout.addWidget(self.cb_density)
        visual_layout.addLayout(viz_flags_layout)

        self.size_label = QLabel("Point Size: 30")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(5, 100)
        self.size_slider.setValue(30)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        visual_layout.addWidget(self.size_label)
        visual_layout.addWidget(self.size_slider)

        self.alpha_label = QLabel("Opacity: 0.80")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(10, 100)
        self.alpha_slider.setValue(80)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        visual_layout.addWidget(self.alpha_label)
        visual_layout.addWidget(self.alpha_slider)

        group_visual.setLayout(visual_layout)
        layout.addWidget(group_visual)

        self.run_btn = QPushButton("🚀 Run Full Pipeline (Ctrl+R)")
        self.run_btn.setStyleSheet("background-color: #4a90e2; color: white; font-weight: bold; padding: 12px; border-radius: 4px;")
        self.run_btn.clicked.connect(self._update_plot)
        layout.addWidget(self.run_btn)

        layout.addStretch()
        self._update_ui_visibility()

    def _update_ui_visibility(self):
        is_text = self.current_data_type in ["Built-in: Text (AI vs Nature)", "Upload: CSV/JSON (Text)", "Upload: TXT (Line-by-Line Text)"]
        is_upload = self.current_data_type.startswith("Upload")
        is_image = "Images" in self.current_data_type
        
        self.group_embed.setEnabled(is_text or is_image)
        
        is_freq_text = is_text and self.current_embed_method in ["TF-IDF", "Bag of Words", "Hashing"]
        self.cb_lower.setEnabled(is_freq_text)
        self.cb_stop.setEnabled(is_freq_text)
        
        show_cols = is_upload and "TXT" not in self.current_data_type
        self.text_col_combo.setVisible(show_cols)
        self.label_col_combo.setVisible(show_cols)
        
        self.upload_btn.setEnabled(is_upload)
        
        model = self.embed_combo.model()
        hog_idx = self.embed_combo.findText("HOG (Image Features)")
        if hog_idx != -1:
            item = model.item(hog_idx)
            if item:
                item.setEnabled(is_image)
                if not is_image and self.current_embed_method == "HOG (Image Features)":
                    self.embed_combo.setCurrentText("TF-IDF")

        is_tsne = (self.current_reduce_method == "t-SNE")
        is_neighbor_based = (self.current_reduce_method in ["UMAP", "Isomap"])
        self.perp_label.setEnabled(is_tsne)
        self.perplexity_slider.setEnabled(is_tsne)
        self.neigh_label.setEnabled(is_neighbor_based)
        self.neigh_slider.setEnabled(is_neighbor_based)

        # Density is only valid for 2D
        self.cb_density.setEnabled(not self.is_3d)

    def _populate_column_dropdowns(self):
        self.text_col_combo.blockSignals(True)
        self.label_col_combo.blockSignals(True)
        self.text_col_combo.clear()
        self.label_col_combo.clear()
        
        if isinstance(self.raw_data, pd.DataFrame):
            cols = list(self.raw_data.columns)
            self.text_col_combo.addItems(cols)
            self.label_col_combo.addItems(["None"] + cols)
        elif isinstance(self.raw_data, list) and self.raw_data and isinstance(self.raw_data[0], dict):
            cols = list(self.raw_data[0].keys())
            self.text_col_combo.addItems(cols)
            self.label_col_combo.addItems(["None"] + cols)
            
        self.text_col_combo.blockSignals(False)
        self.label_col_combo.blockSignals(False)

    def _on_col_changed(self):
        if self.raw_data is not None:
            self._prepare_data()
            self._update_plot()

    def _prepare_data(self):
        try:
            self.legend_info = None
            if self.current_data_type.startswith("Upload"):
                if self.raw_data is None:
                    self.statusBar().showMessage("Please select a file first.")
                    return False
                
                if "TXT" in self.current_data_type:
                    self.data_to_process = self.raw_data
                    self.labels = np.zeros(len(self.raw_data))
                    self.legend_info = {0: "Text"}
                else:
                    df = pd.DataFrame(self.raw_data) if isinstance(self.raw_data, list) else self.raw_data
                    text_col = self.text_col_combo.currentText()
                    label_col = self.label_col_combo.currentText()
                    
                    if not text_col or text_col not in df.columns:
                        return False
                        
                    if "Text" in self.current_data_type:
                        self.data_to_process = df[text_col].astype(str).tolist()
                    else:
                        sample = df[text_col].iloc[0]
                        if isinstance(sample, str) and (sample.startswith('[') or ',' in sample):
                            parsed = df[text_col].apply(lambda s: np.fromstring(s.strip('[]'), sep=','))
                            self.data_to_process = np.vstack(parsed.to_numpy())
                        else:
                            df_num = df.select_dtypes(include=np.number)
                            if df_num.empty:
                                self.statusBar().showMessage("No numeric data found in selected columns.")
                                return False
                            self.data_to_process = df_num.values
                            
                    if label_col == "None" or not label_col or label_col not in df.columns:
                        self.labels = np.zeros(len(df))
                        self.legend_info = {0: "Data"}
                    else:
                        cat = pd.Categorical(df[label_col])
                        self.labels = cat.codes
                        self.legend_info = {i: str(c) for i, c in enumerate(cat.categories)}
                        
            elif self.current_data_type.startswith("Built-in"):
                _, data, labels, legend = self.builtin_data[self.current_data_type]
                self.data_to_process = data
                self.labels = labels
                self.legend_info = legend
                
            return True
        except Exception as e:
            self.statusBar().showMessage(f"Data preparation error: {e}")
            return False

    def _run_pipeline(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(3000)
            
        self.statusBar().showMessage("Processing data...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        
        params = {
            'is_3d': self.is_3d,
            'perplexity': self.perplexity_slider.value(),
            'n_neighbors': self.neigh_slider.value(),
            'labels': self.labels,
            'lowercase': self.cb_lower.isChecked(),
            'remove_stopwords': self.cb_stop.isChecked(),
            'legend_info': self.legend_info
        }
        
        data_type = "Raw Text" if self.current_data_type in [
            "Built-in: Text (AI vs Nature)", "Upload: CSV/JSON (Text)", "Upload: TXT (Line-by-Line Text)"
        ] else "Numeric (Tabular)"
        
        self.worker = PipelineWorker(
            self.data_to_process, data_type, 
            self.current_embed_method, self.current_reduce_method, params
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.error.connect(self._on_pipeline_error)
        self.worker.start()

    def _on_progress(self, msg, val):
        self.statusBar().showMessage(msg)
        self.progress_bar.setValue(val)

    def _on_pipeline_finished(self, X_reduced, y_labels, legend_info, meta):
        self.statusBar().showMessage("Pipeline complete. Hover over points to inspect.")
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self._draw_scatter(X_reduced, y_labels, legend_info, meta)

    def _on_pipeline_error(self, err_msg):
        self.statusBar().showMessage(f"Error: {err_msg}")
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.figure.clear()
        self.canvas.draw_idle()

    def _draw_scatter(self, X_reduced, y_labels, legend_info, meta):
        self.figure.clear()
        self.current_X_reduced = X_reduced
        self.current_y_labels = y_labels
        self.current_meta = meta
        
        unique_labels = np.unique(y_labels)
        cmap = 'tab20' if len(unique_labels) > 10 else 'tab10'
        alpha_val = self.alpha_slider.value() / 100.0
        
        if self.is_3d:
            ax = self.figure.add_subplot(111, projection='3d')
            xs, ys, zs = X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2]
            scatter = ax.scatter(xs, ys, zs, c=y_labels, cmap=cmap, s=self.size_slider.value(), alpha=alpha_val, edgecolors='w', linewidth=0.2)
            ax.set_zlabel("Component 3", color='white')
        else:
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_labels, cmap=cmap, s=self.size_slider.value(), alpha=alpha_val, edgecolors='w', linewidth=0.2)
        
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): 
            spine.set_color('#555555')
            
        embed_str = f"Embedded w/ {self.current_embed_method}" if "Text" in self.current_data_type or "Images" in self.current_data_type else "Raw Numeric Vectors"
        ax.set_title(f"{self.current_data_type}\n{embed_str} -> {self.current_reduce_method} ({'3D' if self.is_3d else '2D'})", color='white', fontsize=12)
        ax.set_xlabel("Component 1", color='white')
        ax.set_ylabel("Component 2", color='white')
        ax.grid(True, linestyle='--', alpha=0.3, color='#555555')

        # --- NEW VIZ FEATURES ---
        
        # 1. Convex Hulls
        if self.cb_hull.isChecked() and not self.is_3d:
            for i in unique_labels:
                points = X_reduced[y_labels == i]
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points[:, :2])
                        for simplex in hull.simplices:
                            ax.plot(points[simplex, 0], points[simplex, 1], 'w-', lw=1.5, alpha=0.4)
                    except Exception:
                        pass # QhullError if points are perfectly collinear

        # 2. Density Contours
        if self.cb_density.isChecked() and not self.is_3d:
            x = X_reduced[:, 0]
            y = X_reduced[:, 1]
            try:
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)
                xmin, xmax = x.min(), x.max()
                ymin, ymax = y.min(), y.max()
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                z_grid = np.reshape(gaussian_kde(xy)(positions).T, xx.shape)
                ax.contour(xx, yy, z_grid, colors='white', alpha=0.3, linewidths=1.5)
            except Exception:
                pass # Fails if dataset is too small or singular

        # 3. PCA Scree Plot Inset
        if meta and 'explained_variance' in meta and not self.is_3d:
            # Create inset axis in top right corner
            ax_inset = self.figure.add_axes([0.65, 0.65, 0.2, 0.2]) # [left, bottom, width, height]
            var_ratio = meta['explained_variance']
            ax_inset.bar(range(1, len(var_ratio)+1), var_ratio, color='#4a90e2', alpha=0.8)
            ax_inset.set_title('PCA Variance', color='white', fontsize=8)
            ax_inset.tick_params(colors='white', labelsize=6)
            ax_inset.set_facecolor('#1e1e1e')
            for spine in ax_inset.spines.values(): 
                spine.set_color('#555555')

        # Build Legend
        if legend_info:
            handles = []
            colors = scatter.cmap(scatter.norm(unique_labels))
            for idx, color in zip(unique_labels, colors):
                label_text = legend_info.get(idx, str(idx))
                handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, linestyle='None', label=label_text))
            ax.legend(handles=handles, title="Classes", facecolor='#1e1e1e', edgecolor='#555555', labelcolor='white')
            ax.get_legend().get_title().set_color("white")
        
        self.hover_annotation = ax.annotate(
            "", xy=(0,0), xytext=(10,10), 
            textcoords="offset points", 
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85), 
            fontsize=9, color='black'
        )
        self.hover_annotation.set_visible(False)
        self.scatter_handle = scatter
        self.canvas.draw_idle()

    def _on_hover(self, event):
        if not hasattr(self, 'scatter_handle') or self.current_X_reduced is None:
            return
        if not self.figure.axes or event.inaxes is not self.figure.axes[0]:
            # Don't hide if hovering over the inset plot
            if not (len(self.figure.axes) > 1 and event.inaxes == self.figure.axes[1]):
                if self.hover_annotation.get_visible():
                    self.hover_annotation.set_visible(False)
                    self.canvas.draw_idle()
            return
            
        cont, ind = self.scatter_handle.contains(event)
        if cont:
            idx = ind["ind"][0]
            x_val = self.current_X_reduced[idx, 0]
            y_val = self.current_X_reduced[idx, 1]
            label_code = self.current_y_labels[idx]
            
            label_text = str(label_code)
            if self.legend_info and label_code in self.legend_info:
                label_text = self.legend_info[label_code]
                
            text = f"Index: {idx}\nLabel: {label_text}\nX: {x_val:.2f}, Y: {y_val:.2f}"
            
            if "Text" in self.current_data_type:
                snippet = str(self.data_to_process[idx])[:60] + "..." if len(str(self.data_to_process[idx])) > 60 else str(self.data_to_process[idx])
                text += f"\nText: {snippet}"
                
            self.hover_annotation.set_text(text)
            self.hover_annotation.xy = (x_val, y_val)
            self.hover_annotation.set_visible(True)
            self.canvas.draw_idle()
        else:
            if self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()

    # --- Event Handlers ---
    def _on_source_changed(self, text):
        self.current_data_type = text
        self._update_ui_visibility()

    def _on_upload_file(self):
        file_filter = "Data Files (*.csv *.json *.txt);;All Files (*)"
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", file_filter)
        if file_name:
            try:
                ext = os.path.splitext(file_name)[1].lower()
                if ext == '.csv':
                    self.raw_data = pd.read_csv(file_name)
                    self._populate_column_dropdowns()
                elif ext == '.json':
                    with open(file_name, 'r', encoding='utf-8') as f:
                        import json
                        data = json.load(f)
                        if isinstance(data, list):
                            self.raw_data = data
                            self._populate_column_dropdowns()
                        else:
                            raise ValueError("JSON must be an array of objects or strings.")
                elif ext == '.txt':
                    with open(file_name, 'r', encoding='utf-8') as f:
                        self.raw_data = [line.strip() for line in f if line.strip()]
                
                self.statusBar().showMessage(f"Loaded {ext.upper()}: {os.path.basename(file_name)}")
                self._prepare_data()
                self._update_plot()
            except Exception as e:
                self.statusBar().showMessage(f"Failed to load file: {e}")
                self.raw_data = None

    def _on_embed_method_changed(self, text):
        self.current_embed_method = text
        self._update_ui_visibility()

    def _on_reduce_method_changed(self, text):
        self.current_reduce_method = text
        self._update_ui_visibility()

    def _on_perplexity_changed(self, val): 
        self.perp_label.setText(f"t-SNE Perplexity: {val}")

    def _on_neighbors_changed(self, val): 
        self.neigh_label.setText(f"Neighbors (UMAP/Isomap): {val}")

    def _on_size_changed(self, val):
        self.size_label.setText(f"Point Size: {val}")
        if hasattr(self, 'scatter_handle') and self.current_X_reduced is not None:
            self.scatter_handle.set_sizes([val] * len(self.current_X_reduced))
            self.canvas.draw_idle()

    def _on_alpha_changed(self, val):
        self.alpha_label.setText(f"Opacity: {val/100.0:.2f}")
        if hasattr(self, 'scatter_handle') and self.current_X_reduced is not None:
            self.scatter_handle.set_alpha(val / 100.0)
            self.canvas.draw_idle()

    def _on_3d_toggle(self, state):
        self.is_3d = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.current_X_reduced = None 
        self._update_ui_visibility()

    def _update_plot(self):
        if self._prepare_data():
            self._run_pipeline()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(60, 60, 60))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    app.setPalette(palette)

    window = UltimateEmbeddingVisualizer()
    window.show()
    sys.exit(app.exec())