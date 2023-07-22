//
//  Compression.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 19/6/23.
//

import Foundation

enum Compression: String, CaseIterable, Identifiable, CustomStringConvertible {
    case quantizied6bit
    case quantizied8bit
    case fullSize
    
    var id: String { rawValue }
    
    var description: String {
        switch self {
        case .quantizied6bit: return "6 bit"
        case .quantizied8bit: return "8 bit"
        case .fullSize: return "Full size"
        }
    }
}
