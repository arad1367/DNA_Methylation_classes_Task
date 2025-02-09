from manim import *
import numpy as np

# Custom colors
DARK_TEAL = "#003333"
GLOW_WHITE = "#E0E0E0"

class DNAMethylationClassification(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = DARK_TEAL

        # Title scene
        self.show_title()
        self.wait(2)

        # Data preprocessing steps
        self.show_data_preprocessing()
        self.wait(2)

        # Dimensionality reduction techniques
        self.show_dimensionality_reduction()
        self.wait(2)

        # Classification models
        self.show_classification_models()
        self.wait(2)

        # Model evaluation
        self.show_model_evaluation()
        self.wait(2)

    def show_title(self):
        title = Text("DNA Methylation Classification of Brain Tumors", font_size=40, color=GLOW_WHITE)
        subtitle = Text("Understanding Epigenetics and Machine Learning", font_size=30, color=GLOW_WHITE)
        subtitle.next_to(title, DOWN)
        self.play(Write(title), Write(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

    def show_data_preprocessing(self):
        title = Text("Data Preprocessing", font_size=35, color=GLOW_WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Steps
        steps = VGroup(
            Text("1. Handle Missing Values", font_size=25),
            Text("2. Normalize Data", font_size=25),
            Text("3. Encode Labels", font_size=25),
            Text("4. Split into Train/Test Sets", font_size=25)
        ).arrange(DOWN, buff=0.5)

        self.play(Write(steps))
        self.wait(2)
        self.play(FadeOut(steps))

    def show_dimensionality_reduction(self):
        title = Text("Dimensionality Reduction", font_size=35, color=GLOW_WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Create 3D scatter plot
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3])
        scatter = VGroup(*[Dot(point=np.random.uniform(-3, 3, 3), color=GLOW_WHITE) for _ in range(100)])
        scatter_label = Text("Sample Distribution", font_size=20).next_to(axes, UP)

        self.play(Create(axes), Create(scatter), Write(scatter_label))
        self.wait(2)
        self.play(FadeOut(axes), FadeOut(scatter), FadeOut(scatter_label))

        # Dimensionality reduction techniques
        techniques = VGroup(
            Text("UMAP", font_size=25),
            Text("t-SNE", font_size=25),
            Text("PCA", font_size=25)
        ).arrange(RIGHT, buff=1)

        self.play(Write(techniques))
        self.wait(2)
        self.play(FadeOut(techniques))

    def show_classification_models(self):
        title = Text("Classification Models", font_size=35, color=GLOW_WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        models = VGroup(
            Text("XGBoost", font_size=25),
            Text("Random Forest", font_size=25),
            Text("Logistic Regression", font_size=25),
            Text("SVM", font_size=25)
        ).arrange(DOWN, buff=0.5)

        self.play(Write(models))
        self.wait(2)
        self.play(FadeOut(models))

    def show_model_evaluation(self):
        title = Text("Model Evaluation", font_size=35, color=GLOW_WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # ROC curve
        axes = Axes(x_range=[0, 1], y_range=[0, 1], axis_config={"color": GLOW_WHITE})
        roc_curve = axes.plot(lambda x: x**2, color=BLUE)
        roc_label = Text("ROC Curve", font_size=20).next_to(axes, UP)

        self.play(Create(axes), Create(roc_curve), Write(roc_label))
        self.wait(2)
        self.play(FadeOut(axes), FadeOut(roc_curve), FadeOut(roc_label))

        # Confusion matrix
        matrix = VGroup(
            Text("TP", font_size=20),
            Text("FP", font_size=20),
            Text("FN", font_size=20),
            Text("TN", font_size=20)
        ).arrange_in_grid(rows=2, cols=2, buff=0.5)
        matrix_label = Text("Confusion Matrix", font_size=20).next_to(matrix, UP)

        self.play(Write(matrix), Write(matrix_label))
        self.wait(2)
        self.play(FadeOut(matrix), FadeOut(matrix_label))

    def show_epigenetics(self):
        title = Text("Epigenetics and DNA Methylation", font_size=35, color=GLOW_WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # DNA strand with methylation
        dna = Line(LEFT * 3, RIGHT * 3, color=GLOW_WHITE)
        methyl_groups = VGroup(*[Dot(point=dna.point_from_proportion(p), color=BLUE) for p in np.linspace(0.1, 0.9, 5)])
        dna_label = Text("DNA Methylation", font_size=20).next_to(dna, UP)

        self.play(Create(dna), Create(methyl_groups), Write(dna_label))
        self.wait(2)
        self.play(FadeOut(dna), FadeOut(methyl_groups), FadeOut(dna_label))

        # Normal vs cancerous methylation
        normal = Text("Normal Methylation", font_size=25)
        cancer = Text("Cancerous Methylation", font_size=25)
        comparison = VGroup(normal, cancer).arrange(RIGHT, buff=2)

        self.play(Write(comparison))
        self.wait(2)
        self.play(FadeOut(comparison))

# Run the animation
if __name__ == "__main__":
    scene = DNAMethylationClassification()
    scene.render()