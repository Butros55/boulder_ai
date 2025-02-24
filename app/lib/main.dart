import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'screens/home_screen.dart';
import 'screens/result_screen.dart';
import 'screens/login_screen.dart';
import 'screens/register_screen.dart';

void main() {
  runApp(const BoulderAi());
}

class BoulderAi extends StatelessWidget {
  const BoulderAi({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Boulder AI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const AuthWrapper(),
        '/login': (context) => const LoginScreen(),
        '/register': (context) => const RegisterScreen(),
        '/home': (context) => const HomeScreen(),
        '/result': (context) {
          final args = ModalRoute.of(context)?.settings.arguments;
          if (args == null || args is! Map<String, dynamic>) {
            return const HomeScreen();
          }
          return ResultScreen(processedResult: args);
        },
      },
    );
  }
}

class AuthWrapper extends StatefulWidget {
  const AuthWrapper({super.key});

  @override
  AuthWrapperState createState() => AuthWrapperState();
}

class AuthWrapperState extends State<AuthWrapper> {
  final storage = const FlutterSecureStorage();
  String? _jwtToken;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _checkToken();
  }

  Future<void> _checkToken() async {
    final token = await storage.read(key: 'jwt_token');
    setState(() {
      _jwtToken = token;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (_jwtToken != null) {
      return const HomeScreen();
    } else {
      return const LoginScreen();
    }
  }
}
