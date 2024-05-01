import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from 'react-native';
import axios from 'axios'
import { useNavigation } from '@react-navigation/native';

const LoginPage = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const navigation = useNavigation();

    const handleLogin = async () => {
        try {
            const response = await axios.post('http://10.0.3.61:5000/login', {
                "email": email,
                "password": password
            });

            console.log(response.data);

            if (response.data.status) {
                navigation.navigate('homePage');
            } else {
                setError(response.data.message);
            }
        } catch (error) {
            console.error('Error logging in:', error);
            setError('An error occurred. Please try again later.');
        }
    };

    const handleRegister = () => {
        // Navigate to the registration page
        navigation.navigate('RegistrationPage');
    };

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Login</Text>
            <TextInput
                style={styles.input}
                placeholder="Email"
                value={email}
                onChangeText={setEmail}
                keyboardType="email-address"
                autoCapitalize="none"
            />
            <TextInput
                style={styles.input}
                placeholder="Password"
                value={password}
                onChangeText={setPassword}
                secureTextEntry
            />
            {error ? <Text style={styles.error}>{error}</Text> : null}
            <TouchableOpacity style={styles.button} onPress={handleLogin}>
                <Text style={styles.buttonText}>Login</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.registerButton} onPress={handleRegister}>
                <Text style={styles.registerButtonText}>Register</Text>
            </TouchableOpacity>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: 20,
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 20,
    },
    input: {
        width: '100%',
        height: 40,
        borderWidth: 1,
        borderColor: '#ccc',
        borderRadius: 5,
        paddingHorizontal: 10,
        marginBottom: 20,
    },
    button: {
        width: '100%',
        height: 40,
        backgroundColor: 'blue',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 5,
        marginBottom: 10,
    },
    buttonText: {
        color: 'white',
        fontWeight: 'bold',
    },
    registerButton: {
        width: '100%',
        height: 40,
        backgroundColor: 'green',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 5,
    },
    registerButtonText: {
        color: 'white',
        fontWeight: 'bold',
    },
    error: {
        color: 'red',
        marginBottom: 10,
    },
});

export default LoginPage;