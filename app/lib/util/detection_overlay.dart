import 'dart:typed_data';
import 'package:flutter/material.dart';

class DetectionOverlayInteractive extends StatelessWidget {
  final Uint8List imageBytes;
  final List<dynamic> detections;
  final double originalWidth;
  final double originalHeight;
  final Map<int, Color> classColorMap;
  final Map<String, int> detectionToRoute;
  final int? selectedRouteIndex;
  final ValueChanged<int> onRouteSelected;

  const DetectionOverlayInteractive({
    Key? key,
    required this.imageBytes,
    required this.detections,
    required this.originalWidth,
    required this.originalHeight,
    required this.classColorMap,
    required this.detectionToRoute,
    required this.selectedRouteIndex,
    required this.onRouteSelected,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final double scaleX = constraints.maxWidth / originalWidth;
        final double scaleY = constraints.maxHeight / originalHeight;

        return Stack(
          children: [
            ClipRRect(
              borderRadius: const BorderRadius.only(bottomLeft: Radius.circular(12), bottomRight: Radius.circular(12)),
              child: Image.memory(imageBytes, width: constraints.maxWidth, height: constraints.maxHeight, fit: BoxFit.cover),
            ),
            // Für jede Erkennung erstellen wir ein Overlay:
            for (var det in detections) _buildOverlay(det, scaleX, scaleY),
          ],
        );
      },
    );
  }

  Widget _buildOverlay(dynamic det, double scaleX, double scaleY) {
    final int clsId = det["class"];
    final Color color = classColorMap[clsId] ?? Colors.white;
    final int? routeIndex = detectionToRoute[det["bbox"].toString()];
    final double alpha = (selectedRouteIndex != null && routeIndex != selectedRouteIndex) ? 0.3 : 1.0;
    final double borderWidth = (selectedRouteIndex != null && routeIndex == selectedRouteIndex) ? 3 : 2;

    if (det.containsKey("segmentation")) {
      final List<dynamic> segPoints = det["segmentation"];
      final List<Offset> points =
          segPoints.map<Offset>((pt) {
            final double x = (pt[0] as num).toDouble() * scaleX;
            final double y = (pt[1] as num).toDouble() * scaleY;
            return Offset(x, y);
          }).toList();

      return Positioned.fill(
        child: GestureDetector(
          onTap: () {
            if (routeIndex != null) onRouteSelected(routeIndex);
          },
          child: CustomPaint(painter: _SegmentationPainter(polygon: points, color: color, borderWidth: borderWidth, opacity: alpha)),
        ),
      );
    } else {
      final List<dynamic> bbox = det["bbox"];
      final double x1 = (bbox[0] as num).toDouble();
      final double y1 = (bbox[1] as num).toDouble();
      final double x2 = (bbox[2] as num).toDouble();
      final double y2 = (bbox[3] as num).toDouble();

      final double left = x1 * scaleX;
      final double top = y1 * scaleY;
      final double width = (x2 - x1) * scaleX;
      final double height = (y2 - y1) * scaleY;

      return Positioned(
        left: left,
        top: top,
        width: width,
        height: height,
        child: GestureDetector(
          onTap: () {
            if (routeIndex != null) onRouteSelected(routeIndex);
          },
          child: Opacity(opacity: alpha, child: Container(decoration: BoxDecoration(border: Border.all(color: color, width: borderWidth)))),
        ),
      );
    }
  }
}

class _SegmentationPainter extends CustomPainter {
  final List<Offset> polygon;
  final Color color;
  final double borderWidth;
  final double opacity;

  _SegmentationPainter({required this.polygon, required this.color, required this.borderWidth, required this.opacity});

  @override
  void paint(Canvas canvas, Size size) {
    if (polygon.isEmpty) return;
    final paint =
        Paint()
          ..color = color.withOpacity(opacity)
          ..strokeWidth = borderWidth
          ..style = PaintingStyle.stroke;
    final path = Path()..moveTo(polygon.first.dx, polygon.first.dy);
    for (var point in polygon.skip(1)) {
      path.lineTo(point.dx, point.dy);
    }
    path.close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _SegmentationPainter oldDelegate) {
    return oldDelegate.polygon != polygon ||
        oldDelegate.color != color ||
        oldDelegate.borderWidth != borderWidth ||
        oldDelegate.opacity != opacity;
  }
}
