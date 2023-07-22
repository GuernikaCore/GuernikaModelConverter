//
//  GuernikaModelConverterApp.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 19/3/23.
//

import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct GuernikaModelConverterApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }.commands {
            SidebarCommands()
            
            CommandGroup(replacing: CommandGroupPlacement.newItem) {}
        }
    }
}
