//
//  Logger.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 30/3/23.
//

import SwiftUI

class Logger: ObservableObject {
    static var shared: Logger = Logger()
    
    enum LogLevel {
        case debug
        case info
        case warning
        case error
        case success
        
        var backgroundColor: Color {
            switch self {
            case .debug:
                return .secondary
            case .info:
                return .blue
            case .warning:
                return .orange
            case .error:
                return .red
            case .success:
                return .green
            }
        }
    }
    
    var isEmpty: Bool = true
    var previousContent: Text?
    @Published var content: Text = Text("")
    
    func append(_ line: String) {
        if line.starts(with: "INFO:") {
            append(String(line.replacing(try! Regex(#"INFO:.*:"#), with: "")), level: .info)
        } else if line.starts(with: "WARNING:") {
            append(String(line.replacing(try! Regex(#"WARNING:.*:"#), with: "")), level: .warning)
        } else if line.starts(with: "ERROR:") {
            append(String(line.replacing(try! Regex(#"ERROR:.*:"#), with: "")), level: .error)
        } else {
            append(line, level: .debug)
        }
    }
    
    func append(_ line: String, level: LogLevel) {
        if level == .success {
            if previousContent == nil {
                previousContent = content
            }
            content = previousContent! + Text(line + "\n").foregroundColor(level.backgroundColor)
            isEmpty = false
            print(line)
        } else {
            previousContent = nil
            if !line.isEmpty {
                content = content + Text(line + "\n").foregroundColor(level.backgroundColor)
                isEmpty = false
                print(line)
            }
        }
    }
    
    func clear() {
        content = Text("")
        previousContent = nil
        isEmpty = true
    }
}
