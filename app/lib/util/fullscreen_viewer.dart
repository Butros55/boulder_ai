import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Vollbild-Dialog mit zoombarem Inhalt.
///
/// - Mausrad- und Pinch-Zoom (bis 12x), Pan per Ziehen
/// - Doppelklick: hineinzoomen bzw. Zoom zurücksetzen
/// - Esc oder X schließt den Dialog
/// - [footer] liegt über dem Inhalt und zoomt nicht mit (z. B. Chip-Leiste)
class FullscreenViewer extends StatefulWidget {
  final String title;

  /// Der zoombare Inhalt. Overlays, die mitzoomen sollen, gehören hier hinein.
  final Widget child;

  /// Zusätzliche Aktionen in der Kopfleiste (z. B. Overlay-Umschalter).
  final List<Widget> actions;

  /// Nicht mitzoomender Bereich am unteren Rand.
  final Widget? footer;

  const FullscreenViewer({
    super.key,
    required this.title,
    required this.child,
    this.actions = const [],
    this.footer,
  });

  @override
  State<FullscreenViewer> createState() => _FullscreenViewerState();
}

class _FullscreenViewerState extends State<FullscreenViewer> with SingleTickerProviderStateMixin {
  final TransformationController _transform = TransformationController();
  late final AnimationController _animCtrl;
  Animation<Matrix4>? _anim;
  Offset? _doubleTapPos;

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 220))
      ..addListener(() {
        final Animation<Matrix4>? anim = _anim;
        if (anim != null) {
          _transform.value = anim.value;
        }
      });
  }

  @override
  void dispose() {
    _animCtrl.dispose();
    _transform.dispose();
    super.dispose();
  }

  void _animateTo(Matrix4 target) {
    _anim = Matrix4Tween(begin: _transform.value, end: target)
        .animate(CurvedAnimation(parent: _animCtrl, curve: Curves.easeOutCubic));
    _animCtrl.forward(from: 0);
  }

  void _resetZoom() => _animateTo(Matrix4.identity());

  void _handleDoubleTap() {
    if (_transform.value.getMaxScaleOnAxis() > 1.05) {
      _resetZoom();
      return;
    }
    final Offset? pos = _doubleTapPos;
    if (pos == null) {
      return;
    }
    const double zoom = 2.5;
    final Matrix4 target = Matrix4.identity()
      ..translateByDouble(-pos.dx * (zoom - 1), -pos.dy * (zoom - 1), 0, 1)
      ..scaleByDouble(zoom, zoom, zoom, 1);
    _animateTo(target);
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      autofocus: true,
      onKeyEvent: (node, event) {
        if (event is KeyDownEvent && event.logicalKey == LogicalKeyboardKey.escape) {
          Navigator.of(context).pop();
          return KeyEventResult.handled;
        }
        return KeyEventResult.ignored;
      },
      child: Dialog.fullscreen(
        backgroundColor: const Color(0xFF131315),
        child: Column(
          children: [
            Container(
              color: const Color(0xFF1C1C1E),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: Row(
                children: [
                  const SizedBox(width: 4),
                  Text(
                    widget.title,
                    style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Text(
                      'Mausrad: Zoom · Ziehen: Verschieben · Doppelklick: Zurücksetzen · Esc: Schließen',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(color: Colors.white.withValues(alpha: 0.38), fontSize: 12),
                    ),
                  ),
                  const SizedBox(width: 8),
                  ...widget.actions,
                  IconButton(
                    tooltip: 'Zoom zurücksetzen',
                    onPressed: _resetZoom,
                    icon: Icon(Icons.zoom_out_map, color: Colors.white.withValues(alpha: 0.7), size: 20),
                  ),
                  IconButton(
                    tooltip: 'Schließen (Esc)',
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.close, color: Colors.white, size: 20),
                  ),
                ],
              ),
            ),
            Expanded(
              child: Stack(
                children: [
                  Positioned.fill(
                    child: GestureDetector(
                      onDoubleTapDown: (details) => _doubleTapPos = details.localPosition,
                      onDoubleTap: _handleDoubleTap,
                      child: InteractiveViewer(
                        transformationController: _transform,
                        minScale: 1,
                        maxScale: 12,
                        child: widget.child,
                      ),
                    ),
                  ),
                  if (widget.footer != null)
                    Positioned(left: 0, right: 0, bottom: 0, child: widget.footer!),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
