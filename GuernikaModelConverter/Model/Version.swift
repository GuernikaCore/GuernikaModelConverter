//
//  Version.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 14/3/23.
//

import Foundation

public struct Version: Comparable, Hashable, CustomStringConvertible {
    let components: [Int]
    public var major: Int {
        guard components.count > 0 else { return 0 }
        return components[0]
    }
    public var minor: Int {
        guard components.count > 1 else { return 0 }
        return components[1]
    }
    public var patch: Int {
        guard components.count > 2 else { return 0 }
        return components[2]
    }
    
    public var description: String {
        return components.map { $0.description }.joined(separator: ".")
    }
    
    public init(major: Int, minor: Int, patch: Int) {
        self.components = [major, minor, patch]
    }
    
    public static func == (lhs: Version, rhs: Version) -> Bool {
        return lhs.major == rhs.major &&
            lhs.minor == rhs.minor &&
            lhs.patch == rhs.patch
    }
    
    public static func < (lhs: Version, rhs: Version) -> Bool {
        return lhs.major < rhs.major ||
            (lhs.major == rhs.major && lhs.minor < rhs.minor) ||
            (lhs.major == rhs.major && lhs.minor == rhs.minor && lhs.patch < rhs.patch)
    }
}

extension Version: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let stringValue = try container.decode(String.self)
        self.init(stringLiteral: stringValue)
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(description)
    }
}

extension Version: ExpressibleByStringLiteral {
    public typealias StringLiteralType = String
    
    public init(stringLiteral value: StringLiteralType) {
        components = value.split(separator: ".").compactMap { Int($0) }
    }
}
