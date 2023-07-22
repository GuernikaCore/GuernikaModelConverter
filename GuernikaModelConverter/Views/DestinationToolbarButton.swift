//
//  DestinationToolbarButton.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 31/3/23.
//

import SwiftUI

struct DestinationToolbarButton: View {
    @Binding var showOutputPicker: Bool
    var outputLocation: URL?
    
    var body: some View {
        ZStack(alignment: .leading) {
            Image(systemName: "folder")
                .padding(.leading, 18)
                .frame(width: 16)
                .foregroundColor(.secondary)
                .onTapGesture {
                    guard let outputLocation else { return }
                    NSWorkspace.shared.open(outputLocation)
                }
            Button {
                showOutputPicker = true
            } label: {
                Text(outputLocation?.lastPathComponent ?? "Select destination")
                    .frame(minWidth: 200)
            }
            .foregroundColor(.primary)
            .background(Color.primary.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .padding(.leading, 34)
        }.background {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.secondary, lineWidth: 1)
                .opacity(0.4)
        }
        .help("Destination")
        .padding(.trailing, 8)
    }
}

struct DestinationToolbarButton_Previews: PreviewProvider {
    static var previews: some View {
        DestinationToolbarButton(showOutputPicker: .constant(false), outputLocation: nil)
    }
}
