//
//  DecimalField.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 5/2/23.
//

import SwiftUI

extension Formatter {
    static let decimal: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.usesGroupingSeparator = false
        formatter.maximumFractionDigits = 2
        return formatter
    }()
}

struct LabeledDecimalField<Content: View>: View {
    @Binding var value: Double
    var step: Double = 1
    var minValue: Double?
    var maxValue: Double?
    @ViewBuilder var label: () -> Content
    
    init(
        _ titleKey: LocalizedStringKey,
        value: Binding<Double>,
        step: Double = 1,
        minValue: Double? = nil,
        maxValue: Double? = nil
    ) where Content == Text {
        self.init(value: value, step: step, minValue: minValue, maxValue: maxValue, label: {
            Text(titleKey)
        })
    }
    
    init(
        value: Binding<Double>,
        step: Double = 1,
        minValue: Double? = nil,
        maxValue: Double? = nil,
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
            DecimalField(
                value: $value,
                step: step,
                minValue: minValue,
                maxValue: maxValue
            )
        }, label: label)
    }
}

struct DecimalField: View {
    @Binding var value: Double
    var step: Double = 1
    var minValue: Double?
    var maxValue: Double?
    @State private var text: String
    @FocusState private var isFocused: Bool
    
    init(
        value: Binding<Double>,
        step: Double = 1,
        minValue: Double? = nil,
        maxValue: Double? = nil
    ) {
        self.step = step
        self.minValue = minValue
        self.maxValue = maxValue
        self._value = value
        let text = Formatter.decimal.string(from: value.wrappedValue as NSNumber) ?? ""
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
                .keyboardType(.decimalPad)
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
            if let newValue = Formatter.decimal.number(from: text)?.doubleValue, value != newValue {
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
        if let newValue = Formatter.decimal.number(from: text)?.doubleValue {
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
        text = Formatter.decimal.string(from: value as NSNumber) ?? ""
    }
}


struct DecimalField_Previews: PreviewProvider {
    static var previews: some View {
        DecimalField(value: .constant(20))
            .padding()
    }
}
