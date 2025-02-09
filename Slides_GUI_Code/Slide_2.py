from manim import *
import numpy as np

class DNAMethylationAnalysis(Scene):
    def construct(self):
        # Title Scene
        self.show_title()
        self.wait(2)
        self.clear_scene()

        # Dataset Overview
        self.show_dataset_overview()
        self.wait(2)
        self.clear_scene()

        # Methylation Class Distinction
        self.show_methylation_heatmap()
        self.wait(2)
        self.clear_scene()

        # Statistical Analysis
        self.show_statistical_analysis()
        self.wait(2)
        self.clear_scene()

        # Machine Learning Approach
        self.show_pca_plot()
        self.wait(2)
        self.clear_scene()

        self.show_roc_curve()
        self.wait(2)
        self.clear_scene()

    def show_title(self):
        title = Text("DNA Methylation Analysis for Histological Label Matching", font_size=40, color=BLUE)
        subtitle = Text("Exploring GSE218542 Dataset", font_size=30, color=WHITE)
        subtitle.next_to(title, DOWN)
        self.play(Write(title), Write(subtitle))
        self.wait(2)

    def clear_scene(self):
        self.play(FadeOut(*self.mobjects))

    def show_dataset_overview(self):
        title = Text("Dataset Overview", font_size=35, color=BLUE)
        description = Text(
            "GSE218542: 86 samples from GEO database\n"
            "Methylation data for histological label matching",
            font_size=25, line_spacing=1.5
        )
        description.next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(description))
        self.wait(2)

    def show_methylation_heatmap(self):
        title = Text("Methylation Class Distinction", font_size=35, color=BLUE)
        description = Text(
            "Heatmap of methylation levels across samples and genes",
            font_size=25
        )
        description.next_to(title, DOWN)

        # Simulate a heatmap
        heatmap = self.create_heatmap(rows=10, cols=10)
        heatmap.next_to(description, DOWN)

        self.play(Write(title))
        self.play(FadeIn(description))
        self.play(Create(heatmap))
        self.wait(2)

    def create_heatmap(self, rows, cols):
        heatmap = VGroup()
        for i in range(rows):
            for j in range(cols):
                value = np.random.uniform(0, 1)
                color = interpolate_color(BLUE, RED, value)
                square = Square(side_length=0.5, fill_color=color, fill_opacity=1, stroke_width=0)
                square.move_to([j - cols/2, rows/2 - i, 0])
                heatmap.add(square)
        return heatmap

    def show_statistical_analysis(self):
        title = Text("Statistical Analysis", font_size=35, color=BLUE)
        description = Text(
            "Hypothesis testing and confidence intervals\n"
            "for methylation differences",
            font_size=25, line_spacing=1.5
        )
        description.next_to(title, DOWN)

        # Simulate a bar chart for statistical results
        bars = self.create_bar_chart([0.8, 0.6, 0.4, 0.2], labels=["Group A", "Group B", "Group C", "Group D"])
        bars.next_to(description, DOWN)

        self.play(Write(title))
        self.play(FadeIn(description))
        self.play(Create(bars))
        self.wait(2)

    def create_bar_chart(self, values, labels):
        chart = VGroup()
        for i, (value, label) in enumerate(zip(values, labels)):
            bar = Rectangle(height=value, width=0.5, fill_color=BLUE, fill_opacity=1, stroke_width=0)
            bar.move_to([i - len(values)/2, value/2, 0])
            label_text = Text(label, font_size=16).next_to(bar, DOWN)
            chart.add(bar, label_text)
        return chart

    def show_pca_plot(self):
        title = Text("Principal Component Analysis (PCA)", font_size=35, color=BLUE)
        description = Text(
            "Visualizing sample clustering based on methylation patterns",
            font_size=25
        )
        description.next_to(title, DOWN)

        # Simulate a PCA plot
        pca_plot = self.create_pca_plot()
        pca_plot.next_to(description, DOWN)

        self.play(Write(title))
        self.play(FadeIn(description))
        self.play(Create(pca_plot))
        self.wait(2)

    def create_pca_plot(self):
        plot = VGroup()
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"color": WHITE})
        points = [Dot(point=[np.random.normal(0, 1), np.random.normal(0, 1), 0], color=BLUE) for _ in range(50)]
        plot.add(axes, *points)
        return plot

    def show_roc_curve(self):
        title = Text("Model Evaluation: ROC Curve", font_size=35, color=BLUE)
        description = Text(
            "Receiver Operating Characteristic (ROC) curve\n"
            "for classification model evaluation",
            font_size=25, line_spacing=1.5
        )
        description.next_to(title, DOWN)

        # Simulate an ROC curve
        roc_curve = self.create_roc_curve()
        roc_curve.next_to(description, DOWN)

        self.play(Write(title))
        self.play(FadeIn(description))
        self.play(Create(roc_curve))
        self.wait(2)

    def create_roc_curve(self):
        curve = VGroup()
        axes = Axes(x_range=[0, 1], y_range=[0, 1], axis_config={"color": WHITE})
        curve_line = axes.plot(lambda x: np.sqrt(x), color=GREEN)
        curve.add(axes, curve_line)
        return curve

# Run the animation
if __name__ == "__main__":
    scene = DNAMethylationAnalysis()
    scene.render()