import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';

/// Ein Schritt einer Zugfolge (Beta) für eine Route.
///
/// [hold] ist der Index innerhalb der zugehörigen Route (`routes[i]`).
/// [hand] ist "left", "right" oder "both"; [action] ist "start", "move",
/// "match" (Doppeln) oder "top" (Ausstieg).
class BetaStep {
  final int step;
  final int hold;
  final String hand;
  final String action;

  /// Optionaler deutscher Kurzhinweis der API (z. B. "Doppeln vor weitem
  /// Zug"). null bei älteren Daten ohne dieses Feld.
  final String? hinweis;

  const BetaStep({
    required this.step,
    required this.hold,
    required this.hand,
    required this.action,
    this.hinweis,
  });
}

/// Parst das Top-Level-Feld `betas` der API-Antwort in eine Liste parallel
/// zu den Routen. Ältere gespeicherte Analysen haben kein `betas`-Feld;
/// dann (und bei jedem Parse-Problem) enthält die Liste nur null-Einträge.
List<List<BetaStep>?> parseBetas(dynamic raw, int routeCount) {
  final List<List<BetaStep>?> result = List<List<BetaStep>?>.filled(routeCount, null);
  dynamic data = raw;
  if (data is String) {
    try {
      data = jsonDecode(data);
    } catch (_) {
      data = null;
    }
  }
  if (data is! List) {
    return result;
  }
  for (int i = 0; i < routeCount && i < data.length; i++) {
    final dynamic rawSteps = data[i];
    if (rawSteps is! List) {
      continue;
    }
    final steps = <BetaStep>[];
    for (final s in rawSteps) {
      if (s is! Map) {
        continue;
      }
      final dynamic step = s["step"];
      final dynamic hold = s["hold"];
      if (step is! num || hold is! num) {
        continue;
      }
      final dynamic hinweis = s["hinweis"];
      steps.add(
        BetaStep(
          step: step.toInt(),
          hold: hold.toInt(),
          hand: s["hand"] is String ? s["hand"] as String : "right",
          action: s["action"] is String ? s["action"] as String : "move",
          hinweis: hinweis is String && hinweis.trim().isNotEmpty ? hinweis.trim() : null,
        ),
      );
    }
    if (steps.isEmpty) {
      continue;
    }
    // Stabil nach Schrittnummer sortieren, damit Startgriffpaare
    // (gleiche step-Nummer) ihre Reihenfolge behalten.
    final List<int> order = List<int>.generate(steps.length, (j) => j);
    order.sort((a, b) {
      final int cmp = steps[a].step.compareTo(steps[b].step);
      return cmp != 0 ? cmp : a.compareTo(b);
    });
    result[i] = [for (final j in order) steps[j]];
  }
  return result;
}

/// Liest die tatsächliche Grifffarbe einer Detektion aus `color_hex`
/// (Fallback: `color.hex`, danach die Klassen-Farbtabelle).
Color resolveDetectionColor(Map det, Map<int, Color> classColorMap) {
  final dynamic hex = det["color_hex"] ?? (det["color"] is Map ? det["color"]["hex"] : null);
  if (hex is String) {
    final String cleaned = hex.replaceFirst('#', '');
    final int? parsed = int.tryParse(cleaned, radix: 16);
    if (parsed != null && cleaned.length == 6) {
      return Color(0xFF000000 | parsed);
    }
  }
  final int clsId = det["class"] is int ? det["class"] as int : 0;
  return classColorMap[clsId] ?? Colors.white;
}

/// Zeigt das Wandbild mit dem Detektions-Overlay:
/// Segmentierungs-Polygone (gefüllt) bzw. abgerundete Boxen, Routenlinien,
/// Routen-Nummern und Interaktion (Griff antippen, leere Fläche antippen).
///
/// Das Bild wird mit BoxFit.contain gerendert. Das Overlay rechnet mit
/// demselben einheitlichen Skalierungsfaktor für x und y plus dem
/// Zentrierungs-Offset (Letterboxing), damit Formen exakt auf den Griffen
/// liegen.
class DetectionOverlayInteractive extends StatelessWidget {
  final Uint8List imageBytes;
  final List<dynamic> detections;

  /// Routen aus der API-Antwort: Liste von Listen von Detection-Dicts.
  final List<dynamic> routes;

  /// Zugfolgen (Beta) parallel zu [routes]; Einträge dürfen null sein
  /// (ältere Analysen ohne `betas`-Feld).
  final List<List<BetaStep>?> betas;

  /// true: für die ausgewählte Route die Zugfolge (Pfeile + Schritt-Badges)
  /// statt der einfachen Routenlinie zeichnen.
  final bool showBeta;
  final double originalWidth;
  final double originalHeight;
  final Map<int, Color> classColorMap;
  final Map<String, int> detectionToRoute;
  final int? selectedRouteIndex;

  /// Overlay komplett ein-/ausblenden.
  final bool showOverlay;

  /// true: Segmentierungs-Polygone bevorzugen, false: immer Bounding-Boxen.
  final bool useShapes;

  /// Detektion, die gerade im Info-Panel angezeigt wird. Sie selbst wird am
  /// stärksten hervorgehoben; gehört sie zu einer Route, leuchtet zusätzlich
  /// die komplette Route (kräftigere Kontur, etwas mehr Füllung).
  final Map? highlightedDetection;

  /// Griff wurde angetippt (liefert Detektion und Routen-Index, falls vorhanden).
  final void Function(Map det, int? routeIndex)? onGripTapped;

  /// Tippen auf eine Fläche ohne Griff.
  final VoidCallback? onEmptyTapped;

  const DetectionOverlayInteractive({
    super.key,
    required this.imageBytes,
    required this.detections,
    this.routes = const [],
    this.betas = const [],
    this.showBeta = false,
    required this.originalWidth,
    required this.originalHeight,
    required this.classColorMap,
    required this.detectionToRoute,
    this.selectedRouteIndex,
    this.showOverlay = true,
    this.useShapes = true,
    this.highlightedDetection,
    this.onGripTapped,
    this.onEmptyTapped,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Einheitlicher Skalierungsfaktor (contain) plus Zentrierungs-Offset.
        final double scale = math.min(
          constraints.maxWidth / originalWidth,
          constraints.maxHeight / originalHeight,
        );
        final Offset origin = Offset(
          (constraints.maxWidth - originalWidth * scale) / 2,
          (constraints.maxHeight - originalHeight * scale) / 2,
        );
        final List<_OverlayItem> items = _prepareItems();
        final List<_RouteLine> routeLines = _prepareRouteLines();
        final _BetaOverlay? beta = _prepareBeta();
        final String? highlightedKey = highlightedDetection?["bbox"]?.toString();

        return ClipRect(
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTapUp: (details) => _handleTap(details.localPosition, items, scale, origin),
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.memory(imageBytes, fit: BoxFit.contain, gaplessPlayback: true),
                if (showOverlay)
                  CustomPaint(
                    painter: _DetectionPainter(
                      items: items,
                      routeLines: routeLines,
                      beta: beta,
                      scale: scale,
                      origin: origin,
                      selectedRouteIndex: selectedRouteIndex,
                      useShapes: useShapes,
                      highlightedKey: highlightedKey,
                      highlightedRouteIndex:
                          highlightedKey != null ? detectionToRoute[highlightedKey] : null,
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  List<_OverlayItem> _prepareItems() {
    final items = <_OverlayItem>[];
    for (final det in detections) {
      if (det is! Map) {
        continue;
      }
      final dynamic bbox = det["bbox"];
      if (bbox is! List || bbox.length < 4) {
        continue;
      }
      final Rect rect = Rect.fromLTRB(
        (bbox[0] as num).toDouble(),
        (bbox[1] as num).toDouble(),
        (bbox[2] as num).toDouble(),
        (bbox[3] as num).toDouble(),
      );
      List<Offset>? polygon;
      final dynamic seg = det["segmentation"];
      if (seg is List && seg.length >= 3) {
        polygon = [
          for (final pt in seg) Offset((pt[0] as num).toDouble(), (pt[1] as num).toDouble()),
        ];
      }
      items.add(
        _OverlayItem(
          det: det,
          key: bbox.toString(),
          color: resolveDetectionColor(det, classColorMap),
          routeIndex: detectionToRoute[bbox.toString()],
          isVolume: det["det_class"] == "volume",
          polygon: polygon,
          bbox: rect,
        ),
      );
    }
    return items;
  }

  List<_RouteLine> _prepareRouteLines() {
    final lines = <_RouteLine>[];
    for (int i = 0; i < routes.length; i++) {
      final dynamic route = routes[i];
      if (route is! List || route.isEmpty) {
        continue;
      }
      final centers = <Offset>[];
      Color? color;
      for (final grip in route) {
        if (grip is! Map) {
          continue;
        }
        final dynamic bbox = grip["bbox"];
        if (bbox is! List || bbox.length < 4) {
          continue;
        }
        color ??= resolveDetectionColor(grip, classColorMap);
        centers.add(
          Offset(
            (((bbox[0] as num) + (bbox[2] as num)) / 2).toDouble(),
            (((bbox[1] as num) + (bbox[3] as num)) / 2).toDouble(),
          ),
        );
      }
      if (centers.isEmpty) {
        continue;
      }
      // Von unten (größtes y) nach oben sortieren, dann als Kette verbinden.
      centers.sort((a, b) => b.dy.compareTo(a.dy));
      lines.add(_RouteLine(index: i, color: color ?? Colors.white, centers: centers));
    }
    return lines;
  }

  /// Bereitet die Zugfolge der ausgewählten Route auf (Pfeile + Badges in
  /// Originalkoordinaten). null, wenn gerade keine Beta gezeichnet wird.
  _BetaOverlay? _prepareBeta() {
    final int? sel = selectedRouteIndex;
    if (!showBeta || sel == null || sel < 0 || sel >= betas.length || sel >= routes.length) {
      return null;
    }
    final List<BetaStep>? steps = betas[sel];
    final dynamic route = routes[sel];
    if (steps == null || steps.isEmpty || route is! List) {
      return null;
    }

    Offset? centerOf(int hold) {
      if (hold < 0 || hold >= route.length) {
        return null;
      }
      final dynamic grip = route[hold];
      if (grip is! Map) {
        return null;
      }
      final dynamic bbox = grip["bbox"];
      if (bbox is! List || bbox.length < 4) {
        return null;
      }
      return Offset(
        (((bbox[0] as num) + (bbox[2] as num)) / 2).toDouble(),
        (((bbox[1] as num) + (bbox[3] as num)) / 2).toDouble(),
      );
    }

    Color color = Colors.white;
    for (final grip in route) {
      if (grip is Map) {
        color = resolveDetectionColor(grip, classColorMap);
        break;
      }
    }

    // Pfeile: je Hand die tatsächliche Bewegung vom letzten Griff der Hand
    // zum neuen Griff (Start setzt nur die Position, ohne Pfeil).
    final arrows = <_BetaArrow>[];
    final seen = <String>{};
    final Map<String, int> lastHold = {};
    for (final s in steps) {
      final List<String> hands = s.hand == 'both' ? const ['left', 'right'] : [s.hand];
      if (s.action != 'start') {
        for (final hand in hands) {
          final int? prev = lastHold[hand];
          if (prev == null || prev == s.hold) {
            continue;
          }
          final Offset? from = centerOf(prev);
          final Offset? to = centerOf(s.hold);
          if (from == null || to == null || !seen.add('$prev>${s.hold}')) {
            continue;
          }
          arrows.add(_BetaArrow(from: from, to: to));
        }
      }
      for (final hand in hands) {
        lastHold[hand] = s.hold;
      }
    }

    // Badges je Griff; mehrere Schritte auf demselben Griff werden
    // untereinander gestapelt, damit sie sich nicht überdecken.
    final badges = <_BetaBadge>[];
    final Map<int, int> perHold = {};
    for (final s in steps) {
      final Offset? anchor = centerOf(s.hold);
      if (anchor == null) {
        continue;
      }
      final int stackIndex = perHold[s.hold] ?? 0;
      perHold[s.hold] = stackIndex + 1;
      badges.add(_BetaBadge(anchor: anchor, stackIndex: stackIndex, step: s));
    }
    if (badges.isEmpty) {
      return null;
    }
    return _BetaOverlay(color: color, arrows: arrows, badges: badges);
  }

  void _handleTap(Offset pos, List<_OverlayItem> items, double scale, Offset origin) {
    if (!showOverlay || scale <= 0) {
      onEmptyTapped?.call();
      return;
    }
    // Tap-Position in Original-Bildkoordinaten umrechnen.
    final Offset p = Offset((pos.dx - origin.dx) / scale, (pos.dy - origin.dy) / scale);
    final double minHit = 16 / scale; // 16 Bildschirm-Pixel Mindest-Trefferfläche
    for (final item in items.reversed) {
      bool hit = false;
      if (useShapes && item.polygon != null) {
        final Path path = Path()..addPolygon(item.polygon!, true);
        hit = path.contains(p);
      }
      if (!hit) {
        Rect box = item.bbox;
        if (box.width < minHit || box.height < minHit) {
          box = Rect.fromCenter(
            center: box.center,
            width: math.max(box.width, minHit),
            height: math.max(box.height, minHit),
          );
        }
        // Bei Polygon-Darstellung zählt die Box nur als Fallback für sehr
        // kleine Griffe, sonst wäre das Klicken unnötig fummelig.
        if (!(useShapes && item.polygon != null) ||
            box.width <= minHit * 1.5 ||
            box.height <= minHit * 1.5) {
          hit = box.contains(p);
        }
      }
      if (hit) {
        onGripTapped?.call(item.det, item.routeIndex);
        return;
      }
    }
    onEmptyTapped?.call();
  }
}

class _OverlayItem {
  final Map det;
  final String key;
  final Color color;
  final int? routeIndex;
  final bool isVolume;
  final List<Offset>? polygon;
  final Rect bbox;

  const _OverlayItem({
    required this.det,
    required this.key,
    required this.color,
    required this.routeIndex,
    required this.isVolume,
    required this.polygon,
    required this.bbox,
  });
}

class _RouteLine {
  final int index;
  final Color color;

  /// Griffmittelpunkte in Originalkoordinaten, von unten nach oben sortiert.
  final List<Offset> centers;

  const _RouteLine({required this.index, required this.color, required this.centers});
}

/// Aufbereitete Zugfolge der ausgewählten Route (alles in Originalkoordinaten).
class _BetaOverlay {
  final Color color;
  final List<_BetaArrow> arrows;
  final List<_BetaBadge> badges;

  const _BetaOverlay({required this.color, required this.arrows, required this.badges});
}

class _BetaArrow {
  final Offset from;
  final Offset to;

  const _BetaArrow({required this.from, required this.to});
}

class _BetaBadge {
  final Offset anchor;

  /// 0 = direkt am Griffmittelpunkt, >0 = darunter gestapelt.
  final int stackIndex;
  final BetaStep step;

  const _BetaBadge({required this.anchor, required this.stackIndex, required this.step});
}

class _DetectionPainter extends CustomPainter {
  final List<_OverlayItem> items;
  final List<_RouteLine> routeLines;

  /// Zugfolge der ausgewählten Route; null = normale Routenlinie zeichnen.
  final _BetaOverlay? beta;

  /// Einheitlicher Skalierungsfaktor für x und y (BoxFit.contain).
  final double scale;

  /// Verschiebung des Bildrechtecks durch das Letterboxing.
  final Offset origin;
  final int? selectedRouteIndex;
  final bool useShapes;
  final String? highlightedKey;

  /// Route der hervorgehobenen Detektion: alle Mitglieder leuchten mit
  /// (kräftigere Kontur, etwas mehr Füllung). null = kein Routen-Leuchten.
  final int? highlightedRouteIndex;

  _DetectionPainter({
    required this.items,
    required this.routeLines,
    required this.beta,
    required this.scale,
    required this.origin,
    required this.selectedRouteIndex,
    required this.useShapes,
    required this.highlightedKey,
    required this.highlightedRouteIndex,
  });

  Offset _mapPoint(Offset p) => Offset(p.dx * scale + origin.dx, p.dy * scale + origin.dy);

  Rect _mapRect(Rect r) => Rect.fromLTRB(
        r.left * scale + origin.dx,
        r.top * scale + origin.dy,
        r.right * scale + origin.dx,
        r.bottom * scale + origin.dy,
      );

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Routenlinien (bzw. Beta-Pfeile) unter die Griff-Formen.
    _paintRouteLines(canvas);
    if (beta != null) {
      _paintBetaArrows(canvas);
    }

    // 2. Gedimmte Detektionen zuerst, aktive darüber, dann die leuchtende
    // Route, der konkret angetippte Griff zuletzt.
    final dimmed = <_OverlayItem>[];
    final active = <_OverlayItem>[];
    final routeGlow = <_OverlayItem>[];
    _OverlayItem? highlighted;
    for (final item in items) {
      if (highlightedKey != null && item.key == highlightedKey) {
        highlighted = item;
        continue;
      }
      if (_isDimmed(item)) {
        dimmed.add(item);
      } else if (highlightedRouteIndex != null && item.routeIndex == highlightedRouteIndex) {
        routeGlow.add(item);
      } else {
        active.add(item);
      }
    }
    for (final item in dimmed) {
      _paintItem(canvas, item, dimmed: true, routeHighlighted: false, highlighted: false);
    }
    for (final item in active) {
      _paintItem(canvas, item, dimmed: false, routeHighlighted: false, highlighted: false);
    }
    for (final item in routeGlow) {
      _paintItem(canvas, item, dimmed: false, routeHighlighted: true, highlighted: false);
    }
    if (highlighted != null) {
      final bool isDimmed = _isDimmed(highlighted);
      _paintItem(
        canvas,
        highlighted,
        dimmed: isDimmed,
        routeHighlighted:
            !isDimmed && highlightedRouteIndex != null && highlighted.routeIndex == highlightedRouteIndex,
        highlighted: true,
      );
    }

    // 3. Routen-Nummern bzw. Schritt-Badges oben drauf, damit sie lesbar bleiben.
    _paintRouteBadges(canvas);
    if (beta != null) {
      _paintBetaBadges(canvas);
    }
  }

  bool _isDimmed(_OverlayItem item) {
    return selectedRouteIndex != null && item.routeIndex != selectedRouteIndex;
  }

  void _paintRouteLines(Canvas canvas) {
    for (final line in routeLines) {
      // Bei aktiver Beta ersetzt die Zugfolge die Linie der gewählten Route.
      if (beta != null && line.index == selectedRouteIndex) {
        continue;
      }
      if (line.centers.length < 2) {
        continue;
      }
      final double? alpha = _routeVisibility(line.index, dezent: 0.5, kraeftig: 0.9);
      if (alpha == null) {
        continue;
      }
      final Paint paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round
        ..color = line.color.withValues(alpha: alpha);
      final Offset start = _mapPoint(line.centers.first);
      final Path path = Path()..moveTo(start.dx, start.dy);
      for (final c in line.centers.skip(1)) {
        final Offset mapped = _mapPoint(c);
        path.lineTo(mapped.dx, mapped.dy);
      }
      canvas.drawPath(path, paint);
    }
  }

  void _paintRouteBadges(Canvas canvas) {
    for (final line in routeLines) {
      // Bei aktiver Beta übernehmen die Schritt-Badges die Beschriftung.
      if (beta != null && line.index == selectedRouteIndex) {
        continue;
      }
      final double? alpha = _routeVisibility(line.index, dezent: 0.85, kraeftig: 1);
      if (alpha == null) {
        continue;
      }
      final Offset center = _mapPoint(line.centers.first);
      const double radius = 9;
      canvas.drawCircle(center, radius, Paint()..color = line.color.withValues(alpha: alpha));
      canvas.drawCircle(
        center,
        radius,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5
          ..color = Colors.white.withValues(alpha: 0.9 * alpha),
      );
      final Color numberColor = line.color.computeLuminance() > 0.55 ? Colors.black : Colors.white;
      final TextPainter tp = TextPainter(
        text: TextSpan(
          text: '${line.index + 1}',
          style: TextStyle(
            color: numberColor.withValues(alpha: alpha),
            fontSize: 10,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, center - Offset(tp.width / 2, tp.height / 2));
    }
  }

  /// null = Route ist gerade nicht sichtbar.
  double? _routeVisibility(int routeIndex, {required double dezent, required double kraeftig}) {
    if (selectedRouteIndex == null) {
      return dezent;
    }
    if (selectedRouteIndex == routeIndex) {
      return kraeftig;
    }
    return null;
  }

  void _paintBetaArrows(Canvas canvas) {
    final _BetaOverlay data = beta!;
    final Paint linePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round
      ..color = data.color.withValues(alpha: 0.9);
    final Paint headPaint = Paint()..color = data.color.withValues(alpha: 0.95);

    for (final arrow in data.arrows) {
      final Offset from = _mapPoint(arrow.from);
      final Offset to = _mapPoint(arrow.to);
      final Offset delta = to - from;
      final double dist = delta.distance;
      if (dist < 4) {
        continue;
      }
      final Offset dir = delta / dist;
      // An beiden Enden kürzen, damit die Badges an den Griffen frei bleiben.
      final Offset start = from + dir * math.min(10, dist * 0.25);
      final Offset tip = to - dir * math.min(12, dist * 0.25);
      final double headLength = math.min(8, dist * 0.3);
      final Offset lineEnd = tip - dir * (headLength * 0.7);
      canvas.drawLine(start, lineEnd, linePaint);

      final Offset perp = Offset(-dir.dy, dir.dx);
      final Offset base = tip - dir * headLength;
      final Path head = Path()
        ..moveTo(tip.dx, tip.dy)
        ..lineTo(base.dx + perp.dx * headLength * 0.55, base.dy + perp.dy * headLength * 0.55)
        ..lineTo(base.dx - perp.dx * headLength * 0.55, base.dy - perp.dy * headLength * 0.55)
        ..close();
      canvas.drawPath(head, headPaint);
    }
  }

  void _paintBetaBadges(Canvas canvas) {
    final _BetaOverlay data = beta!;
    for (final badge in data.badges) {
      _paintBetaBadge(canvas, badge, data.color);
    }
  }

  /// Kleiner Chip am Schritt-Griff: Schrittnummer (+ "S" beim Start bzw.
  /// "TOP" beim Ausstieg) und Hand-Symbol(e). Heller Grund, dunkle Schrift.
  void _paintBetaBadge(Canvas canvas, _BetaBadge badge, Color routeColor) {
    const Color textColor = Color(0xFF1C1C1E);
    final BetaStep s = badge.step;
    final String label = s.action == 'top' ? 'TOP' : (s.action == 'start' ? '${s.step} S' : '${s.step}');
    final TextPainter tp = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: textColor,
          fontSize: 9,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.3,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    const double iconSize = 10;
    const double padH = 5;
    const double gap = 3;
    const double height = 16;
    const double hintDotRadius = 2;
    final bool leftHand = s.hand == 'left' || s.hand == 'both';
    final bool rightHand = s.hand == 'right' || s.hand == 'both';
    final bool hasHint = s.hinweis != null;
    final int iconCount = (leftHand ? 1 : 0) + (rightHand ? 1 : 0);
    final double width = padH * 2 +
        tp.width +
        (iconCount > 0 ? gap + iconCount * iconSize + (iconCount - 1) : 0) +
        (hasHint ? gap + hintDotRadius * 2 : 0);

    final Offset center = _mapPoint(badge.anchor) + Offset(0, badge.stackIndex * (height + 3));
    final Rect rect = Rect.fromCenter(center: center, width: width, height: height);
    final RRect rr = RRect.fromRectAndRadius(rect, const Radius.circular(8));
    canvas.drawRRect(rr, Paint()..color = Colors.white.withValues(alpha: 0.95));
    canvas.drawRRect(
      rr,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.2
        ..color = routeColor.withValues(alpha: 0.95),
    );

    double x = rect.left + padH;
    tp.paint(canvas, Offset(x, center.dy - tp.height / 2));
    x += tp.width + gap;
    if (leftHand) {
      _paintHandIcon(canvas, x, center.dy, iconSize, textColor, mirrored: true);
      x += iconSize + (rightHand ? 1 : 0);
    }
    if (rightHand) {
      _paintHandIcon(canvas, x, center.dy, iconSize, textColor, mirrored: false);
      x += iconSize;
    }
    // Kleiner Punkt als Indikator, dass zu diesem Schritt ein Hinweis
    // existiert (voller Text steht in der Zugfolge-Karte der Seitenleiste).
    if (hasHint) {
      canvas.drawCircle(
        Offset(x + gap + hintDotRadius, center.dy),
        hintDotRadius,
        Paint()..color = textColor.withValues(alpha: 0.7),
      );
    }
  }

  /// Hand-Symbol (Material Icon) zeichnen; für die linke Hand gespiegelt.
  void _paintHandIcon(
    Canvas canvas,
    double left,
    double centerY,
    double size,
    Color color, {
    required bool mirrored,
  }) {
    final TextPainter tp = TextPainter(
      text: TextSpan(
        text: String.fromCharCode(Icons.back_hand.codePoint),
        style: TextStyle(
          fontFamily: Icons.back_hand.fontFamily,
          package: Icons.back_hand.fontPackage,
          fontSize: size,
          color: color,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final Offset pos = Offset(left, centerY - tp.height / 2);
    if (!mirrored) {
      tp.paint(canvas, pos);
      return;
    }
    final double axis = left + tp.width / 2;
    canvas.save();
    canvas.translate(axis, 0);
    canvas.scale(-1, 1);
    canvas.translate(-axis, 0);
    tp.paint(canvas, pos);
    canvas.restore();
  }

  void _paintItem(
    Canvas canvas,
    _OverlayItem item, {
    required bool dimmed,
    required bool routeHighlighted,
    required bool highlighted,
  }) {
    final double strokeAlpha = dimmed ? 0.12 : 1;
    final bool inSelectedRoute = selectedRouteIndex != null && item.routeIndex == selectedRouteIndex;
    double fillAlpha = dimmed ? 0.04 : 0.25;
    double strokeWidth = inSelectedRoute ? 3 : 2;
    if (routeHighlighted) {
      // Die ganze Route leuchtet mit dem angetippten Griff mit.
      strokeWidth += 1.5;
      fillAlpha = 0.35;
    }
    if (highlighted) {
      // Der konkret angetippte Griff (bzw. Schritt-Griff) nochmal stärker.
      strokeWidth = math.max(strokeWidth + 0.5, 3.5);
    }

    Path outline;
    bool isPolygon = false;
    if (useShapes && item.polygon != null) {
      outline = Path()..addPolygon([for (final p in item.polygon!) _mapPoint(p)], true);
      isPolygon = true;
    } else {
      outline = Path()..addRRect(RRect.fromRectAndRadius(_mapRect(item.bbox), const Radius.circular(6)));
    }

    if (isPolygon) {
      canvas.drawPath(
        outline,
        Paint()
          ..style = PaintingStyle.fill
          ..color = item.color.withValues(alpha: fillAlpha),
      );
    }

    if (highlighted && !dimmed) {
      canvas.drawPath(
        outline,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = strokeWidth + 3
          ..strokeJoin = StrokeJoin.round
          ..color = Colors.white.withValues(alpha: 0.5),
      );
    }

    final Paint strokePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeJoin = StrokeJoin.round
      ..color = item.color.withValues(alpha: strokeAlpha);

    if (item.isVolume) {
      canvas.drawPath(_dashPath(outline), strokePaint);
      if (!dimmed) {
        _paintVolLabel(canvas, item);
      }
    } else {
      canvas.drawPath(outline, strokePaint);
    }
  }

  void _paintVolLabel(Canvas canvas, _OverlayItem item) {
    final Rect rect = _mapRect(item.bbox);
    final Color textColor = item.color.computeLuminance() > 0.55 ? Colors.black : Colors.white;
    final TextPainter tp = TextPainter(
      text: TextSpan(
        text: 'VOL',
        style: TextStyle(color: textColor, fontSize: 9, fontWeight: FontWeight.bold, letterSpacing: 0.6),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final double top = math.max(origin.dy + 1, rect.top - tp.height - 8);
    final Rect bg = Rect.fromLTWH(rect.left, top, tp.width + 10, tp.height + 5);
    canvas.drawRRect(
      RRect.fromRectAndRadius(bg, const Radius.circular(4)),
      Paint()..color = item.color.withValues(alpha: 0.85),
    );
    tp.paint(canvas, Offset(bg.left + 5, bg.top + 2.5));
  }

  Path _dashPath(Path source, {double dashLength = 6, double gapLength = 4}) {
    final Path dest = Path();
    for (final metric in source.computeMetrics()) {
      double distance = 0;
      bool draw = true;
      while (distance < metric.length) {
        final double length = draw ? dashLength : gapLength;
        if (draw) {
          dest.addPath(metric.extractPath(distance, distance + length), Offset.zero);
        }
        distance += length;
        draw = !draw;
      }
    }
    return dest;
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) {
    return oldDelegate.items != items ||
        oldDelegate.routeLines != routeLines ||
        oldDelegate.beta != beta ||
        oldDelegate.scale != scale ||
        oldDelegate.origin != origin ||
        oldDelegate.selectedRouteIndex != selectedRouteIndex ||
        oldDelegate.useShapes != useShapes ||
        oldDelegate.highlightedKey != highlightedKey ||
        oldDelegate.highlightedRouteIndex != highlightedRouteIndex;
  }
}
