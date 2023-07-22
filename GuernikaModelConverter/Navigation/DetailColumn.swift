//
//  DetailColumn.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 19/3/23.
//

import SwiftUI

struct DetailColumn: View {
    @Binding var path: NavigationPath
    @Binding var selection: Panel
    @StateObject var modelConverter = ConvertModelViewModel()
    @StateObject var controlNetConverter = ConvertControlNetViewModel()
    
    var body: some View {
        switch selection {
        case .model:
            ConvertModelView(model: modelConverter)
        case .controlNet:
            ConvertControlNetView(model: controlNetConverter)
        case .log:
            LogView()
        }
    }
}

struct DetailColumn_Previews: PreviewProvider {
    struct Preview: View {
        @State private var selection: Panel = .model
        
        var body: some View {
            DetailColumn(path: .constant(NavigationPath()), selection: $selection)
        }
    }
    static var previews: some View {
        Preview()
    }
}
