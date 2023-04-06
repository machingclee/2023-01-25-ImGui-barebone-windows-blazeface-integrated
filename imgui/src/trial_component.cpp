#include "imgui/trial_component.h"

void TrialComponent::Render() {
    static bool buttonPressed = ImGui::Button("Button");
    if (buttonPressed) {
        print("The button has been pressed");
        _p_open = false;
    }

    static int a = 0;
    ImGui::RadioButton("RadioButton0", &a, 0);
    ImGui::RadioButton("RadioButton1", &a, 1);
    ImGui::RadioButton("RadioButton2", &a, 2);
    // HelpMarker("Please select a device ssociated to your camera.");

    ImGui::Text("This is a text");
    ImGui::BulletText("This is a bullet text");

    static float value_float = 0;
    ImGui::SliderFloat("Float Slider", &value_float, 0.f, 1.f);
    static int value_int = 0;
    ImGui::SliderInt("Int Slider", &value_int, 0, 100);

    static int counter = 0;
    bool counterButtonClicked = ImGui::Button("click me");
    if (counterButtonClicked) {
        counter += 1;
    }
    ImGui::Text("Counter: %i", counter);

    // static int selectedIndex = 0;
    // std::vector<const char*> items = {"item1", "item2"};
    // ImGui::Combo("#ItemsCombo", &selectedIndex, items.data(), items.size());

    ImGui::End();

    static bool showing = false;
    static bool circleChecked = false;

    showing = GetKeyState(VK_DELETE);

    if (showing) {
        ImGui::Begin("test");
        ImGui::Checkbox("#circle fov", &circleChecked);
        ImGui::End();
    }
}