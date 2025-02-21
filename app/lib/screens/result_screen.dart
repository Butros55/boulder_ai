import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> processedResult;

  const ResultScreen({super.key, required this.processedResult});

  @override
  Widget build(BuildContext context) {
    Uint8List origBytes;
    Uint8List filtBytes;
    Uint8List detectBytes;

    try {
      final String origBase64 = processedResult["original_image"] as String;
      final String filtBase64 = processedResult["filtered_image"] as String;
      final String detectBase64 = processedResult["detection_image"] as String;

      origBytes = base64Decode(origBase64);
      filtBytes = base64Decode(filtBase64);
      detectBytes = base64Decode(detectBase64);
    } catch (e) {
      return Scaffold(
        appBar: AppBar(title: const Text('Verarbeitetes Bild')),
        body: Center(child: Text('Fehler beim Dekodieren der Bilder: $e')),
      );
    }

    final detections = processedResult["detections"];
    String jsonString = const JsonEncoder.withIndent('  ').convert(detections);

    return Scaffold(
      appBar: AppBar(title: const Text('Boulder AI Ergebnisse')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildImageContainer(origBytes, 'Original'),
                const SizedBox(width: 16),
                _buildImageContainer(filtBytes, 'Farbfilter'),
                const SizedBox(width: 16),
                _buildImageContainer(detectBytes, 'KI-Detektion'),
              ],
            ),
            const SizedBox(height: 20),
            Text(
              'Erkannte Griffe (YOLO-Detektionen):',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12.0),
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                jsonString,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 14),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageContainer(Uint8List bytes, String label) {
    return Column(
      children: [
        Container(
          width: 300,
          height: 350,
          decoration: BoxDecoration(
            border: Border.all(color: Colors.grey, width: 2),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Image.memory(bytes, fit: BoxFit.contain),
        ),
        const SizedBox(height: 8),
        Text(label),
      ],
    );
  }
}
