//
//  IntegerField.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 23/1/22.
//

import SwiftUI

struct LabeledIntegerField<Content: View>: View {
    @Binding var value: Int
    var step: Int = 1
    var minValue: Int?
    var maxValue: Int?
    @ViewBuilder var label: () -> Content
    
    init(
        _ titleKey: LocalizedStringKey,
        value: Binding<Int>,
        step: Int = 1,
        minValue: Int? = nil,
        maxValue: Int? = nil
    ) where Content == Text {
        self.init(value: value, step: step, minValue: minValue, maxValue: maxValue, label: {
            Text(titleKey)
        })
    }
    
    init(
        value: Binding<Int>,
        step: Int = 1,
        minValue: Int? = nil,
        maxValue: Int? = nil,
        @ViewBuilder label: @escaping () -> Content
    ) {
        self.label = label
        self.step = step
        self.minValue = minValue
        self.maxValue = maxValue
        self._value = value
    }
    
    var body: some View {
        LabeledContent(content: {
            IntegerField(
                value: $value,
                step: step,
                minValue: minValue,
                maxValue: maxValue
            )
            .frame(maxWidth: 120)
        }, label: label)
    }
}

struct IntegerField: View {
    @Binding var value: Int
    var step: Int = 1
    var minValue: Int?
    var maxValue: Int?
    @State private var text: String
    @FocusState private var isFocused: Bool
    
    init(
        value: Binding<Int>,
         step: Int = 1,
         minValue: Int? = nil,
         maxValue: Int? = nil
    ) {
        self.step = step
        self.minValue = minValue
        self.maxValue = maxValue
        self._value = value
        let text = String(describing: value.wrappedValue)
        self._text = State(wrappedValue: text)
    }
    
    var body: some View {
        HStack(spacing: 0) {
            TextField("", text: $text, prompt: Text("Value"))
                .multilineTextAlignment(.trailing)
                .textFieldStyle(.plain)
                .padding(.horizontal, 10)
                .submitLabel(.done)
                .focused($isFocused)
                .frame(minWidth: 70)
                .labelsHidden()
#if !os(macOS)
                .keyboardType(.numberPad)
#endif
            Stepper(label: {}, onIncrement: {
                if let maxValue {
                    value = min(value + step, maxValue)
                } else {
                    value += step
                }
            }, onDecrement: {
                if let minValue {
                    value = max(value - step, minValue)
                } else {
                    value -= step
                }
            }).labelsHidden()
        }
#if os(macOS)
        .padding(3)
        .background(Color.primary.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
#else
        .padding(2)
        .background(Color.primary.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
#endif
        .onSubmit {
            updateValue()
            isFocused = false
        }
        .onChange(of: isFocused) { focused in
            if !focused {
                updateValue()
            }
        }
        .onChange(of: value) { _ in updateText() }
        .onChange(of: text) { _ in
            if let newValue = Int(text), value != newValue {
                value = newValue
            }
        }
#if !os(macOS)
        .toolbar {
            if isFocused {
                ToolbarItem(placement: .keyboard) {
                    HStack {
                        Spacer()
                        Button("Done") {
                            isFocused = false
                        }
                    }
                }
            }
        }
#endif
    }
    
    private func updateValue() {
        if let newValue = Int(text) {
            if let maxValue, newValue > maxValue {
                value = maxValue
            } else if let minValue, newValue < minValue {
                value = minValue
            } else {
                value = newValue
            }
        }
    }
    
    private func updateText() {
        text = String(describing: value)
    }
}

struct ValueField_Previews: PreviewProvider {
    static var previews: some View {
        LabeledIntegerField("Text", value: .constant(20))
    }
}

