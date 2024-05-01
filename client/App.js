import { StatusBar } from 'expo-status-bar';
import { ImageBackground, StyleSheet, Text, View } from 'react-native';
import LandingPage from './src/LandingPage';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomePage from './src/HomePage';
import CameraPage from './src/CameraPage';
import LoginPage from './src/LoginPage';
import RegistrationPage from './src/RegistrationPage';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name='landingPage' component={LandingPage} options={{ headerShown: false }} />
        <Stack.Screen name='loginPage' component={LoginPage} options={{ headerShown: false }} />
        <Stack.Screen name='homePage' component={HomePage} options={{ headerShown: false }} />
        <Stack.Screen name='RegistrationPage' component={RegistrationPage} options={{ headerShown: false }} />
        <Stack.Screen name='cameraPage' component={CameraPage} options={{ headerShown: false }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
