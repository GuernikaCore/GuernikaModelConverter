//
//  ContentView.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 19/3/23.
//

import SwiftUI

struct ContentView: View {
    @State private var selection: Panel = Panel.model
    @State private var path = NavigationPath()
    
    var body: some View {
        NavigationSplitView {
            Sidebar(path: $path, selection: $selection)
        } detail: {
            NavigationStack(path: $path) {
                DetailColumn(path: $path, selection: $selection)
            }
        }
        .onChange(of: selection) { _ in
            path.removeLast(path.count)
        }
        .frame(minWidth: 800, minHeight: 500)
    }
}

struct ContentView_Previews: PreviewProvider {
    struct Preview: View {
        var body: some View {
            ContentView()
        }
    }
    static var previews: some View {
        Preview()
    }
}
