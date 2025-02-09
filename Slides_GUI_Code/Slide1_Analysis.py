from manim import *

class MethylationDataPreparation(Scene):
    def construct(self):
        # Title Scene
        self.show_title()
        self.wait(2)

        # Initial Data Understanding
        self.show_initial_data_structure()
        self.wait(2)

        # Key Challenge: ID Mismatch
        self.show_id_mismatch_challenge()
        self.wait(2)

        # Metadata Structure Discovery
        self.show_metadata_structure()
        self.wait(2)

        # Data Integration Solution
        self.show_id_mapping_and_integration()
        self.wait(2)

        # Data Transformation
        self.show_data_transformation()
        self.wait(2)

        # Final Data Structure and Class Distribution
        self.show_final_data_structure()
        self.wait(2)

    def show_title(self):
        title = Text("Methylation Data Preparation", font_size=40, color=BLUE)
        subtitle = Text("A Step-by-Step Data Story", font_size=30, color=WHITE)
        subtitle.next_to(title, DOWN)
        self.play(Write(title), Write(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

    def show_initial_data_structure(self):
        # Title
        title = Text("Initial Data Understanding", font_size=35, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Show two files side by side
        methylation_file = Text("Methylation Data", font_size=25, color=GREEN)
        metadata_file = Text("Metadata", font_size=25, color=YELLOW)
        files = VGroup(methylation_file, metadata_file).arrange(RIGHT, buff=2)
        self.play(Create(files))
        self.wait(1)

        # Highlight unusual column orientation
        arrow = Arrow(methylation_file.get_bottom(), metadata_file.get_bottom(), color=RED)
        orientation_label = Text("Unusual Column Orientation", font_size=20, color=RED).next_to(arrow, DOWN)
        self.play(Create(arrow), Write(orientation_label))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(files), FadeOut(arrow), FadeOut(orientation_label))

    def show_id_mismatch_challenge(self):
        # Title
        title = Text("Key Challenge: ID Mismatch", font_size=35, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Show array IDs and GSM IDs
        array_ids = Text("Array IDs", font_size=25, color=GREEN)
        gsm_ids = Text("GSM IDs", font_size=25, color=YELLOW)
        ids = VGroup(array_ids, gsm_ids).arrange(RIGHT, buff=2)
        self.play(Create(ids))
        self.wait(1)

        # Highlight mismatch
        mismatch_arrow = Arrow(array_ids.get_bottom(), gsm_ids.get_bottom(), color=RED)
        mismatch_label = Text("Mismatch", font_size=20, color=RED).next_to(mismatch_arrow, DOWN)
        self.play(Create(mismatch_arrow), Write(mismatch_label))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(ids), FadeOut(mismatch_arrow), FadeOut(mismatch_label))

    def show_metadata_structure(self):
        # Title
        title = Text("Metadata Structure Discovery", font_size=35, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Show metadata lines
        line1 = Text("Line 1: Array ID", font_size=20, color=GREEN)
        line2 = Text("Line 2: GSM ID", font_size=20, color=YELLOW)
        line3 = Text("Line 3: Label", font_size=20, color=BLUE)
        lines = VGroup(line1, line2, line3).arrange(DOWN, buff=0.5)
        self.play(Create(lines))
        self.wait(1)

        # Highlight crucial lines
        highlight_box = SurroundingRectangle(VGroup(line1, line2), color=RED)
        highlight_label = Text("Crucial for ID Mapping", font_size=20, color=RED).next_to(highlight_box, DOWN)
        self.play(Create(highlight_box), Write(highlight_label))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(lines), FadeOut(highlight_box), FadeOut(highlight_label))

    def show_id_mapping_and_integration(self):
        # Title
        title = Text("Data Integration Solution", font_size=35, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Show dictionaries
        id_mapping = Text("ID Mapping Dictionary", font_size=25, color=GREEN)
        label_assignment = Text("Label Assignment Dictionary", font_size=25, color=YELLOW)
        dictionaries = VGroup(id_mapping, label_assignment).arrange(DOWN, buff=1)
        self.play(Create(dictionaries))
        self.wait(1)

        # Animate integration
        integration_arrow = Arrow(id_mapping.get_right(), label_assignment.get_right(), color=BLUE)
        integration_label = Text("Integration", font_size=20, color=BLUE).next_to(integration_arrow, RIGHT)
        self.play(Create(integration_arrow), Write(integration_label))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(dictionaries), FadeOut(integration_arrow), FadeOut(integration_label))

    def show_data_transformation(self):
        # Title
        title = Text("Data Transformation", font_size=35, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Show transformation steps
        cleaning = Text("1. Cleaning", font_size=25, color=GREEN)
        transposing = Text("2. Transposing", font_size=25, color=YELLOW)
        labeling = Text("3. Labeling", font_size=25, color=BLUE)
        steps = VGroup(cleaning, transposing, labeling).arrange(DOWN, buff=0.5)
        self.play(Create(steps))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(steps))

    def show_final_data_structure(self):
        # Title
        title = Text("Final Data Structure", font_size=35, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Show final data shape
        data_shape = Text("Shape: (Samples, Features)", font_size=25, color=GREEN)
        class_distribution = Text("Class Distribution: Balanced", font_size=25, color=YELLOW)
        final_data = VGroup(data_shape, class_distribution).arrange(DOWN, buff=1)
        self.play(Create(final_data))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(final_data))

# Run the animation
if __name__ == "__main__":
    scene = MethylationDataPreparation()
    scene.render()