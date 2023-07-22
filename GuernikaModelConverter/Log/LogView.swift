//
//  LogView.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 30/3/23.
//

import SwiftUI

struct LogView: View {
    @ObservedObject var logger = Logger.shared
    @State var stickToBottom: Bool = true
    
    var body: some View {
        VStack {
            if logger.isEmpty {
                emptyView
            } else {
                contentView
            }
        }.navigationTitle("Log")
            .toolbar {
                ToolbarItemGroup {
                    Button {
                        logger.clear()
                    } label: {
                        Image(systemName: "trash")
                    }.help("Clear log")
                        .disabled(logger.isEmpty)
                    Toggle(isOn: $stickToBottom) {
                        Image(systemName: "dock.arrow.down.rectangle")
                    }.help("Stick to bottom")
                }
            }
    }
    
    @ViewBuilder
    var emptyView: some View {
        Image(systemName: "moon.zzz.fill")
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(width: 72)
            .opacity(0.3)
            .padding(8)
        Text("Log is empty")
            .font(.largeTitle)
            .opacity(0.3)
    }
    
    @ViewBuilder
    var contentView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                logger.content
                    .multilineTextAlignment(.leading)
                    .font(.body.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                
                Divider().opacity(0)
                    .id("bottom")
            }.onChange(of: logger.content) { _ in
                if stickToBottom {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }.onChange(of: stickToBottom) { newValue in
                if newValue {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }.onAppear {
                if stickToBottom {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
        }
    }
}

struct LogView_Previews: PreviewProvider {
    static var previews: some View {
        LogView()
    }
}
