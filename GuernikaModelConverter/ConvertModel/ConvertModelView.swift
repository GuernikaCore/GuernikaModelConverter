//
//  ConvertModelView.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 30/3/23.
//

import SwiftUI

struct ConvertModelView: View {
    @ObservedObject var model: ConvertModelViewModel
    
    var body: some View {
        VStack(spacing: 16) {
            if model.isRunning {
                runningView
            } else {
                originPicker
                computeUnitsPicker
                customSizePicker
                precisionFullToggle
                if #available(macOS 14.0, *) {
                    compressionPicker
                }
                modulesGrid
                embeddingsSection
            }
        }
        .padding()
        .navigationTitle("Convert model")
        .toolbar {
            ToolbarItemGroup {
                toolbarItems
            }
        }
        .alert("Error", isPresented: $model.showError, actions: {
            if model.isCoreMLError {
                Button("Install Xcode") {
                    NSWorkspace.shared.open(URL(string:"https://developer.apple.com/xcode/")!)
                }
                Button("Launch Terminal") {
                    NSWorkspace.shared.openApplication(
                        at: URL(string:"file:///System/Applications/Utilities/Terminal.app")!,
                        configuration: NSWorkspace.OpenConfiguration()
                    )
                }
            }
            Button("OK", role: .cancel, action: {
                model.showError = false
            })
        }, message: {
            model.error.map { Text($0) }
        })
        .alert("Success", isPresented: $model.showSuccess, actions: {
            if var outputLocation = model.outputLocation {
                Button("Reveal in Finder", action: {
                    switch model.modelOrigin {
                    case .huggingface:
                        outputLocation.append(components: model.huggingfaceIdentifier.replacingOccurrences(of: "/", with: "_"))
                    case .diffusers:
                        if let diffusersLocation = model.diffusersLocation {
                            outputLocation.append(components: diffusersLocation.lastPathComponent)
                        }
                    case .checkpoint:
                        if let checkpointLocation = model.checkpointLocation {
                            outputLocation.append(components: checkpointLocation.deletingPathExtension().lastPathComponent)
                        }
                    }
                    NSWorkspace.shared.activateFileViewerSelecting([outputLocation])
                    model.showSuccess = false
                })
            }
            Button("OK", role: .cancel, action: {
                model.showSuccess = false
            })
        }, message: {
            Text("Model successfully converted")
        })
    }
    
    @ViewBuilder
    var runningView: some View {
        if let process = model.process {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(process.steps) { step in
                    HStack {
                        if step == process.currentStep {
                            CircularProgress(progress: process.currentProgress?.percentage)
                        } else if step.rawValue < process.currentStep.rawValue {
                            Image(systemName: "checkmark.circle.fill")
                                .resizable()
                                .frame(width: 16, height: 16)
                                .padding(4)
                                .foregroundColor(.green)
                        } else {
                            Image(systemName: "checkmark.circle")
                                .resizable()
                                .frame(width: 16, height: 16)
                                .padding(4)
                                .foregroundColor(.gray)
                        }
                        VStack(alignment: .leading) {
                            Text(step.description)
                                .font(.headline)
                            if step == process.currentStep, let progress = process.currentProgress {
                                Text(progress.description)
                                    .foregroundColor(.secondary)
                                    .font(.caption)
                            }
                        }.multilineTextAlignment(.leading)
                            .frame(minHeight: 36)
                            .frame(width: 380, alignment: .leading)
                    }
                    .font(.headline)
                }
            }
        }
    }
    
    @ViewBuilder
    var originPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model origin")
                .foregroundColor(.secondary)
                .font(.headline)
                .padding(.horizontal, 4)
            Picker("", selection: $model.modelOrigin, content: {
                ForEach(ModelOrigin.allCases) { units in
                    Text(units.description)
                        .tag(units)
                }
            })
            .pickerStyle(.segmented)
            .labelsHidden()
            originExtra
        }.frame(maxWidth: 480)
    }
    
    @State var copyHelp: Bool = false
    @State var droppingModel: Bool = false
    @State var selectingModel: Bool = false
    
    @ViewBuilder
    var originExtra: some View {
        switch model.modelOrigin {
        case .huggingface:
            HStack(alignment: .center) {
                TextField("Model version", text: $model.huggingfaceIdentifier, prompt: Text("Model version"))
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 8)
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.writeObjects([model.huggingfaceIdentifier as NSPasteboardWriting])
                    copyHelp = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + .seconds(1)) {
                        copyHelp = false
                    }
                } label: {
                    Image(systemName: "doc.on.doc")
                }.help("Copy")
                    .disabled(model.huggingfaceIdentifier.isEmpty)
                    .popover(isPresented: $copyHelp) {
                        Text("Copied!").padding(8)
                    }
                Button {
                    NSPasteboard.general.pasteboardItems?.first?.string(forType: .string)
                        .map { model.huggingfaceIdentifier = $0 }
                } label: {
                    Image(systemName: "doc.on.clipboard")
                }.help("Paste")
            }
            .padding(3)
            .background(Color.primary.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        case .diffusers:
            HStack(alignment: .center) {
                if droppingModel {
                    Text("Drop to use as origin")
                        .foregroundColor(.secondary)
                } else if let selectedModel = model.selectedModel {
                    Button {
                        model.diffusersLocation = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }.buttonStyle(.plain)
                        .foregroundColor(.secondary)
                    Text(selectedModel)
                }
                Spacer()
                Button("Select folder") {
                    selectingModel = true
                }.fileImporter(isPresented: $selectingModel, allowedContentTypes: [.folder]) { result in
                    if case .success(let modelUrl) = result {
                        model.diffusersLocation = modelUrl
                    } else {
                        print("File import failed")
                    }
                }
            }
            .frame(minHeight: 24)
            .padding(6)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white.opacity(droppingModel ? 0.03 : 0))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(.secondary, lineWidth: 1)
                            .opacity(droppingModel ? 0.5 : 0)
                    )
            )
            .onDrop(of: [.url], isTargeted: $droppingModel) { providers in
                if let provider = providers.first(where: { $0.canLoadObject(ofClass: URL.self) }) {
                    let _ = provider.loadObject(ofClass: URL.self) { reading, error in
                        if let reading {
                            DispatchQueue.main.async {
                                model.diffusersLocation = reading
                            }
                        }
                        if let error {
                            print("error", error)
                        }
                    }
                    return true
                }
                return false
            }
        case .checkpoint:
            HStack(alignment: .center) {
                if droppingModel {
                    Text("Drop to use as origin")
                        .foregroundColor(.secondary)
                } else if let selectedModel = model.selectedModel {
                    Button {
                        model.checkpointLocation = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }.buttonStyle(.plain)
                        .foregroundColor(.secondary)
                    Text(selectedModel)
                }
                Spacer()
                Button("Select checkpoint") {
                    selectingModel = true
                }.fileImporter(isPresented: $selectingModel, allowedContentTypes: [.item]) { result in
                    if case .success(let checkpointUrl) = result {
                        model.checkpointLocation = checkpointUrl
                    } else {
                        print("File import failed")
                    }
                }
            }
            .frame(minHeight: 24)
            .padding(6)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white.opacity(droppingModel ? 0.03 : 0))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(.secondary, lineWidth: 1)
                            .opacity(droppingModel ? 0.5 : 0)
                    )
            )
            .onDrop(of: [.url], isTargeted: $droppingModel) { providers in
                if let provider = providers.first(where: { $0.canLoadObject(ofClass: URL.self) }) {
                    let _ = provider.loadObject(ofClass: URL.self) { reading, error in
                        if let reading {
                            DispatchQueue.main.async {
                                model.checkpointLocation = reading
                            }
                        }
                        if let error {
                            print("error", error)
                        }
                    }
                    return true
                }
                return false
            }
        }
    }
    
    @ViewBuilder
    var computeUnitsPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Compute units")
                .foregroundColor(.secondary)
                .font(.headline)
                .padding(.horizontal, 4)
            ViewThatFits {
                Picker("", selection: $model.computeUnits, content: {
                    ForEach(ComputeUnits.allCases) { units in
                        Text(units.description)
                            .tag(units)
                    }
                })
                .pickerStyle(.segmented)
                Picker("", selection: $model.computeUnits, content: {
                    ForEach(ComputeUnits.allCases) { units in
                        Text(units.shortDescription)
                            .help(units.description)
                            .tag(units)
                    }
                })
                .pickerStyle(.segmented)
            }
            .frame(maxWidth: 480)
            .labelsHidden()
            Text(model.computeUnits == .cpuAndGPU ? "Original attention will be used" : "Split einsum attention will be used")
                .foregroundColor(.secondary)
                .font(.footnote)
                .padding(.horizontal, 6)
        }
    }
    
    @ViewBuilder
    var modulesGrid: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Modules to convert")
                .foregroundColor(.secondary)
                .font(.headline)
                .padding(.horizontal, 4)
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 12) {
                GridRow {
                    Toggle("Unet", isOn: $model.convertUnet)
                        .frame(maxWidth: 160, alignment: .leading)
                    Toggle("Chunk Unet", isOn: $model.chunkUnet)
                        .disabled(!model.convertUnet)
                        .frame(maxWidth: 160, alignment: .leading)
                    Toggle("ControlNet support", isOn: $model.controlNetSupport)
                        .disabled(!model.convertUnet)
                        .frame(maxWidth: 160, alignment: .leading)
                }
                GridRow {
                    Toggle("Encoder", isOn: $model.convertVaeEncoder)
                    Toggle("Decoder", isOn: $model.convertVaeDecoder)
                    Button("Select all", action: model.selectAll)
                }
                GridRow {
                    Toggle("Text encoder", isOn: $model.convertTextEncoder)
                    Toggle("Safety checker", isOn: $model.convertSafetyChecker)
                    Button("Select none", action: model.selectNone)
                }
            }
        }
        .frame(maxWidth: 480)
    }
    
    @ViewBuilder
    var customSizePicker: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading) {
                Toggle("Custom size", isOn: $model.customSize)
                    .font(.headline)
                    .padding(.horizontal, 4)
                if model.computeUnits != .cpuAndGPU {
                    Label("Only available when using\nCPU and GPU compute units", systemImage: "exclamationmark.triangle")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 5)
                }
            }.multilineTextAlignment(.leading)
            Spacer()
            VStack {
                LabeledIntegerField("Width", value: $model.customWidth, step: 8, minValue: 128)
                LabeledIntegerField("Height", value: $model.customHeight, step: 8, minValue: 128)
            }.disabled(!model.customSize)
                .foregroundColor(model.customSize ? nil : .secondary)
        }
        .frame(maxWidth: 480)
        .disabled(model.computeUnits != .cpuAndGPU)
    }
    
    @ViewBuilder
    var precisionFullToggle: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle("Use full precision", isOn: $model.precisionFull)
                .frame(maxWidth: 480, alignment: .leading)
            if model.computeUnits != .cpuAndGPU {
                Label("Only available when using CPU and GPU compute units", systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 5)
            } else {
                Text("Increases fidelity to the original model but some will also increase the size of the converted model")
                    .font(.footnote)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 5)
            }
        }
        .frame(maxWidth: 480)
        .disabled(model.computeUnits != .cpuAndGPU)
    }
    
    @ViewBuilder
    var compressionPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Compression")
                .foregroundColor(.secondary)
                .font(.headline)
                .padding(.horizontal, 4)
            Picker("", selection: $model.compression, content: {
                ForEach(Compression.allCases) { units in
                    Text(units.description)
                        .tag(units)
                }
            })
            .pickerStyle(.segmented)
            .labelsHidden()
            Text("Compressing the output model reduces significantly the size while resulting in some detail loss.")
                .font(.footnote)
                .foregroundColor(.secondary)
                .padding(.horizontal, 5)
        }
        .frame(maxWidth: 480)
    }
    
    @State var selectingEmbeddingsLocation: Bool = false
    
    @ViewBuilder
    var embeddingsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Textual inversion")
                .foregroundColor(.secondary)
                .font(.headline)
                .padding(.horizontal, 4)
            HStack {
                Toggle("Load text encoder embeddings", isOn: $model.loadEmbeddings)
                Spacer()
                Button("Select folder") {
                    selectingEmbeddingsLocation = true
                }.fileImporter(isPresented: $selectingEmbeddingsLocation, allowedContentTypes: [.folder]) { result in
                    if case .success(let modelUrl) = result {
                        model.embeddingsLocation = modelUrl
                    } else {
                        print("File import failed")
                    }
                }
                .disabled(!model.loadEmbeddings)
            }
            if model.loadEmbeddings && model.convertTextEncoder, let embeddingsLocation = model.embeddingsLocation {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Text encoder will include embeddings at:")
                    Text(embeddingsLocation.path(percentEncoded: false))
                }.foregroundColor(.secondary)
                    .font(.footnote)
                    .padding(.horizontal, 6)
            }
        }
        .frame(maxWidth: 480)
        .disabled(!model.convertTextEncoder)
        .opacity(model.convertTextEncoder ? 1 : 0.4)
    }
    
    @ViewBuilder
    var toolbarItems: some View {
        if model.isRunning {
            Button(role: .cancel, action: {
                model.cancel()
            }, label: {
                Image(systemName: "stop.fill")
            }).keyboardShortcut(KeyEquivalent("."), modifiers: .command)
                .help("Stop conversion")
        } else {
            DestinationToolbarButton(showOutputPicker: $model.showOutputPicker, outputLocation: model.outputLocation)
            
            Button(action: {
                model.start()
            }, label: {
                Image(systemName: "play.fill")
            }).keyboardShortcut(KeyEquivalent("g"), modifiers: [.command])
            .help("Start conversion")
            .disabled(!model.isReady)
            .fileImporter(isPresented: $model.showOutputPicker, allowedContentTypes: [.folder]) { result in
                if case .success(let modelUrl) = result {
                    model.outputLocation = modelUrl
                } else {
                    print("File import failed")
                }
            }
        }
    }
}

struct ConvertModelView_Previews: PreviewProvider {
    static var previews: some View {
        ConvertModelView(model: ConvertModelViewModel())
    }
}
